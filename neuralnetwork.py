import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection      import train_test_split
from sklearn.preprocessing       import StandardScaler
from sklearn.utils.class_weight   import compute_class_weight
from imblearn.over_sampling       import BorderlineSMOTE

import tensorflow as tf
from tensorflow.keras             import Model, Input
from tensorflow.keras.layers      import Dense, Dropout, BatchNormalization, GaussianNoise, LeakyReLU
from tensorflow.keras.callbacks   import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

from sklearn.metrics              import confusion_matrix, classification_report

# Rutas
base_dir = "/Users/danillopis/Desktop/TFG/UAB/workload_dataset"
input_dir = os.path.join(base_dir, "outputs_preproc")
output_dir = os.path.join(base_dir, "outputs_features")
os.makedirs(output_dir, exist_ok=True)

# Cargar features con etiqueta de carga cognitiva
dir_in_cog = os.path.join(base_dir, "outputs_cognitive_load")
os.makedirs(dir_in_cog, exist_ok=True)
features_cog_path = os.path.join(dir_in_cog, "features_cognitive_load.csv")
features_cog_df = pd.read_csv(features_cog_path)

# Eliminar columnas irrelevantes
cols_to_drop = [
    'perceived_difficulty', 'start_time', 'phase', 'flight',
    'composite_index', 'new_cognitive_load_global',
    'subject_median', 'new_cognitive_load_subject',
    'rolling_median', 'new_cognitive_load_rolling', 'subject', 'theoretical_difficulty'
]
X = features_cog_df.drop(columns=cols_to_drop, errors='ignore')
y = features_cog_df['theoretical_difficulty'].to_numpy()

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train / test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
# train -> train/val
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2,
    stratify=y_train, random_state=42
)

# SMOTE en train (BorderlineSMOTE)
sm = BorderlineSMOTE(random_state=42)
X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)

# Pesos de clase manuales (por ejemplo, desequilibrio)
custom_weights = {0: 2.0, 1: 1.0, 2: 2.0}
classes = np.unique(y_tr_res)
cw_vals = compute_class_weight(
    class_weight=custom_weights,
    classes=classes,
    y=y_tr_res
)
class_weight = dict(zip(classes, cw_vals))

print("Distribución antes SMOTE:", pd.Series(y_tr).value_counts().to_dict())
print("Distribución tras SMOTE:", pd.Series(y_tr_res).value_counts().to_dict())
print("Class weights:", class_weight)

# ----------------------------------------
# 3) Definición de Focal Loss y Modelo DNN
# ----------------------------------------

def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=len(classes))
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = - y_true_oh * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1.0 - y_pred, gamma)
        fl = weight * ce
        return tf.reduce_sum(fl, axis=1)
    return loss_fn

def build_model(units_list, input_dim,
                dropout_rate=0.5, l2_reg=1e-3, lr=1e-3):
    inp = Input(shape=(input_dim,))
    x = GaussianNoise(0.1)(inp)
    for units in units_list:
        x = Dense(units,
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(l2_reg))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
    out = Dense(len(classes), activation='softmax')(x)

    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    return model

architectures = {
    'small':   [128,  64],
    'medium':  [256, 128, 64],
    'large':   [512, 256, 64],
    'xlarge':  [512, 256, 128, 64],
    'xxlarge': [1024, 512, 256, 128, 64]
}

histories = {}
results    = {}

# Crear carpeta para salidas de la DNN
dir_out_nn = os.path.join(base_dir, "outputs_nn")
os.makedirs(dir_out_nn, exist_ok=True)

for name, units in architectures.items():
    print(f"\n=== Entrenando {name} net ===")
    model = build_model(
        units,
        input_dim=X_tr_res.shape[1],
        dropout_rate=0.5,
        l2_reg=1e-3,
        lr=1e-3
    )
    model.summary()

    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
    ck_path = os.path.join(dir_out_nn, f'best_{name}.keras')
    ck = ModelCheckpoint(ck_path, monitor='val_loss', save_best_only=True)

    hist = model.fit(
        X_tr_res, y_tr_res,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=64,
        class_weight=class_weight,
        callbacks=[es, rl, ck],
        verbose=2
    )
    histories[name] = hist

    # Evaluación en TEST
    model.load_weights(ck_path)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # Matriz de confusión y heatmap
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n--- {name} net: Matriz de Confusión ---")
    print(cm)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{name} net Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(dir_out_nn, f'{name}_confusion.png'))
    plt.close()

    # Informe de clasificación
    print(f"--- {name} net: Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in classes]))
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Cálculo de recall_macro y spec_macro
    rec_macro = report_dict['macro avg']['recall']
    specs = []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + (cm[i, :].sum() - tp))
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    spec_macro = np.mean(specs)
    results[name] = (rec_macro, spec_macro)

    # Guardar classification report como CSV
    pd.DataFrame(report_dict).to_csv(os.path.join(dir_out_nn, f'{name}_report.csv'))

    print(f"{name:6s} → Recall_macro: {rec_macro:.3f}, Spec_macro: {spec_macro:.3f}")

# ----------------------------------------
# 4) Guardar resultados y curvas de entrenamiento
# ----------------------------------------

# Guardar resultados resumen
df_results = pd.DataFrame(results, index=['recall_macro', 'spec_macro']).T
df_results.to_csv(os.path.join(dir_out_nn, 'nn_results.csv'))

# Plot de curvas de pérdida (loss) para cada arquitectura
import math
n = len(histories)
cols = 2
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
axes = axes.flatten()

for ax, (name, hist) in zip(axes, histories.items()):
    ax.plot(hist.history['loss'],  label='train')
    ax.plot(hist.history['val_loss'],label='val')
    ax.set_title(f'{name} net')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

# Quitar ejes vacíos si sobran
for ax in axes[n:]:
    fig.delaxes(ax)

plt.tight_layout()
plt.savefig(os.path.join(dir_out_nn, 'training_loss.png'))
plt.close()

print("\n=== Entrenamiento y evaluación de redes neuronales completado. Outputs en:", dir_out_nn, "===\n")