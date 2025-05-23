import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import BorderlineSMOTE
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report

# Paths
dir_base = "/Users/danillopis/Desktop/TFG/UAB/workload_dataset"
dir_in = os.path.join(dir_base, "outputs_cognitive_load")
dir_out = os.path.join(dir_base, "outputs_nn")
os.makedirs(dir_out, exist_ok=True)

# Load features with cognitive load label
features_path = os.path.join(dir_in, "features_cognitive_load.csv")
features_df = pd.read_csv(features_path)

# Drop irrelevant columns
cols_to_drop = [
    'perceived_difficulty', 'start_time', 'phase', 'flight',
    'composite_index', 'new_cognitive_load_global',
    'subject_median', 'new_cognitive_load_subject',
    'rolling_median', 'new_cognitive_load_rolling'
]
X = features_df.drop(columns=cols_to_drop, errors='ignore')
y = features_df['theoretical_difficulty'].to_numpy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train/test and train/val
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# SMOTE on training
sm = BorderlineSMOTE(random_state=42)
X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)

# Compute class weights
classes = np.unique(y_tr_res)
cw_vals = compute_class_weight('balanced', classes=classes, y=y_tr_res)
class_weight = dict(zip(classes, cw_vals))

# Save distributions
dist_df = pd.DataFrame({
    'before': pd.Series(y_tr).value_counts(),
    'after': pd.Series(y_tr_res).value_counts()
})
dist_df.to_csv(os.path.join(dir_out, 'smote_distribution.csv'))
pd.Series(class_weight).to_csv(os.path.join(dir_out, 'class_weights.csv'), header=['weight'])

# Focal loss
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=len(classes))
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
        ce = -y_true_oh * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1-y_pred, gamma)
        fl = weight * ce
        return tf.reduce_sum(fl, axis=1)
    return loss_fn

# Model builder
def build_model(units_list, input_dim, dropout=0.4, l2_reg=1e-3, lr=1e-3):
    inp = Input(shape=(input_dim,))
    x = GaussianNoise(0.1)(inp)
    for u in units_list:
        x = Dense(u, kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
    out = Dense(len(classes), activation='softmax')(x)
    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=focal_loss(), metrics=['accuracy']
    )
    return model

# Architectures
architectures = {
    'small':  [128, 64],
    'medium': [256, 128, 64],
    'large':  [512, 256, 128, 64],
    'xlarge': [1024, 512, 256, 128, 64]
}

histories = {}
results = {}

for name, units in architectures.items():
    print(f"Training {name} net...")
    model = build_model(units, input_dim=X_tr_res.shape[1])
    ckpt_path = os.path.join(dir_out, f'best_{name}.keras')
    callbacks = [
        EarlyStopping('val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau('val_loss', factor=0.5, patience=8, min_lr=1e-6),
        ModelCheckpoint(ckpt_path, 'val_loss', save_best_only=True)
    ]
    hist = model.fit(
        X_tr_res, y_tr_res,
        validation_data=(X_val, y_val),
        epochs=300, batch_size=64,
        class_weight=class_weight,
        callbacks=callbacks, verbose=2
    )
    histories[name] = hist
    # Evaluate
    model.load_weights(ckpt_path)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    # Save confusion matrix plot
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} CM')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.savefig(os.path.join(dir_out, f'{name}_confusion.png'))
    plt.close()
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).to_csv(os.path.join(dir_out, f'{name}_report.csv'))
    # Recall & specificity
    recall = report['macro avg']['recall']
    specs = []
    for i in range(cm.shape[0]):
        tp = cm[i,i]; fp = cm[:,i].sum()-tp
        tn = cm.sum()-(tp+fp+(cm[i,:].sum()-tp))
        specs.append(tn/(tn+fp) if (tn+fp)>0 else 0)
    spec = np.mean(specs)
    results[name] = {'recall_macro': recall, 'spec_macro': spec}
    print(f"{name}: recall={recall:.3f}, spec={spec:.3f}")

# Save results summary
df_results = pd.DataFrame(results).T
df_results.to_csv(os.path.join(dir_out, 'nn_results.csv'))

# Plot training curves
plt.figure(figsize=(10,8))
for name, hist in histories.items():
    plt.plot(hist.history['loss'], label=f'{name}_train')
    plt.plot(hist.history['val_loss'], label=f'{name}_val')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(dir_out, 'training_loss.png'))
plt.close()

print('Neural network training and evaluation complete. Outputs in', dir_out)