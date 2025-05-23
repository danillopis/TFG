import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind

# Configuración de rutas
base_dir = "/Users/danillopis/Desktop/TFG/UAB/workload_dataset"
input_dir = os.path.join(base_dir, "outputs_features")
output_dir = os.path.join(base_dir, "outputs_cognitive_load")
os.makedirs(output_dir, exist_ok=True)

# Carga del DataFrame de características
features_path = os.path.join(input_dir, "features.csv")
features_df = pd.read_csv(features_path)

# Asegurar datetime si existe
if 'start_time' in features_df.columns:
    features_df['start_time'] = pd.to_datetime(features_df['start_time'])

# Definir listas de columnas y parámetros
electrode_cols = [c.split('_')[0] for c in features_df.columns if '_' in c]
electrode_cols = sorted(set(electrode_cols))
bands = ['delta','theta','alpha','beta','gamma']

# 1) Selección de columnas de potencia
power_cols = [f"{ch}_{band}_power" for ch in electrode_cols for band in bands if f"{ch}_{band}_power" in features_df.columns]

# Guardar lista de columnas seleccionadas
df_power_cols = pd.DataFrame({'power_column': power_cols})
df_power_cols.to_csv(os.path.join(output_dir, "selected_power_columns.csv"), index=False)

# 2) Estandarización y cálculo de índice compuesto global
X = features_df[power_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
composite_index_global = X_scaled.mean(axis=1)

# Visualizar y guardar histograma del índice global
plt.figure(figsize=(10,4))
plt.hist(composite_index_global, bins=30, edgecolor='k')
plt.title('Distribución Índice Compuesto - Potencia')
plt.xlabel('Índice Compuesto')
plt.ylabel('Frecuencia')
hist_path = os.path.join(output_dir, 'hist_index_power.png')
plt.tight_layout()
plt.savefig(hist_path)
plt.close()

# Definir umbral global y variable binaria
global_threshold = np.median(composite_index_global)
features_df['cognitive_load_global'] = (composite_index_global > global_threshold).astype(int)
print(f"Umbral global: {global_threshold:.4f}")

# 3) Índice compuesto combinado (temporal + frecuencia)
# Columnas temporales: numéricas que no contienen 'power'
temporal_cols = [col for col in features_df.columns if np.issubdtype(features_df[col].dtype, np.number) and 'power' not in col and col not in ['theoretical_difficulty','start_time','subject']]
combined_cols = temporal_cols + power_cols
X_combined = features_df[combined_cols].values
X_combined_scaled = scaler.fit_transform(X_combined)
composite_index_combined = X_combined_scaled.mean(axis=1)

# Histograma índice combinado
plt.figure(figsize=(10,4))
plt.hist(composite_index_combined, bins=30, edgecolor='k')
plt.title('Distribución Índice Compuesto - Combinado')
plt.xlabel('Índice Compuesto')
plt.ylabel('Frecuencia')
plt.savefig(os.path.join(output_dir, 'hist_index_combined.png'))
plt.close()

# Crear variable binaria para carga combinada
combined_threshold = np.median(composite_index_combined)
features_df['cognitive_load_combined'] = (composite_index_combined > combined_threshold).astype(int)
print(f"Umbral combinado: {combined_threshold:.4f}")

# 4) Índice personalizado por sujeto
df_subjects = []
for subj in features_df['subject'].unique():
    mask = features_df['subject'] == subj
    subj_index = composite_index_combined[mask]
    subj_thresh = np.median(subj_index)
    features_df.loc[mask, 'cognitive_load_subject'] = (subj_index > subj_thresh).astype(int)
    df_subjects.append((subj, subj_thresh))
# Guardar umbrales por sujeto
df_subjects = pd.DataFrame(df_subjects, columns=['subject','threshold'])
df_subjects.to_csv(os.path.join(output_dir,'thresholds_by_subject.csv'), index=False)

# 5) Variable con rolling median
tmp_df = pd.DataFrame({'composite_index': composite_index_combined})
tmp_df['rolling_median'] = tmp_df['composite_index'].rolling(window=50, min_periods=1).median()
tmp_df['cognitive_load_rolling'] = (tmp_df['composite_index'] > tmp_df['rolling_median']).astype(int)
features_df['cognitive_load_rolling'] = tmp_df['cognitive_load_rolling']

# Guardar DataFrame final con nuevas variables
dest_path = os.path.join(output_dir, 'features_cognitive_load.csv')
features_df.to_csv(dest_path, index=False)
print("Variables de carga cognitiva añadidas. CSV guardado en:", dest_path)
