import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import iirnotch, butter, filtfilt, decimate
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
import mne
from mne.preprocessing import ICA
from scipy.stats import kurtosis
import matplotlib.pyplot as plt


# Configuración de rutas
dir_base = "/Users/danillopis/Desktop/TFG/UAB/workload_dataset"
dir_in = os.path.join(dir_base, "outputs_eda")
dir_out = os.path.join(dir_base, "outputs_preproc")
os.makedirs(dir_out, exist_ok=True)

# Carga de datasets del paso de EDA
df_eeg = pd.read_csv(os.path.join(dir_in, "df_eeg_eda.csv"))
df_ecg_hr = pd.read_csv(os.path.join(dir_in, "df_ecg_hr_eda.csv"))
df_ecg_ibi = pd.read_csv(os.path.join(dir_in, "df_ecg_ibi_eda.csv"))

# Copias locales
eeg = df_eeg.copy()
ecg_hr = df_ecg_hr.copy()
ecg_ibi = df_ecg_ibi.copy()

# Asegurar datetime
eeg['datetime'] = pd.to_datetime(eeg['datetime'])
ecg_hr['datetime'] = pd.to_datetime(ecg_hr['datetime'])
ecg_ibi['datetime'] = pd.to_datetime(ecg_ibi['datetime'])

# 1) Eliminación de duplicados en ECG HR
keys = ['subject', 'flight', 'phase', 'datetime']
agg_hr = {col: ('mean' if col=='hr' else 'first') for col in ecg_hr.columns if col not in keys}
df_ecg_hr_no_dups = ecg_hr.groupby(keys, as_index=False).agg(agg_hr)

# 2) Eliminación de duplicados en ECG IBI
agg_ibi = {col: ('mean' if col=='rr_int' else 'first') for col in ecg_ibi.columns if col not in keys}
df_ecg_ibi_no_dups = ecg_ibi.groupby(keys, as_index=False).agg(agg_ibi)

# 3) Merge ECG
ecg = df_ecg_hr_no_dups.merge(
    df_ecg_ibi_no_dups[keys + ['rr_int']], on=keys, how='left'
)

# 4) Señal EEG brute raw para un canal ejemplo
electrode_cols = [
    'EEG.AF3','EEG.F7','EEG.F3','EEG.FC5',
    'EEG.T7','EEG.P7','EEG.O1','EEG.O2',
    'EEG.P8','EEG.T8','EEG.FC6','EEG.F4',
    'EEG.F8','EEG.AF4'
]
df_eeg_temp = eeg[['datetime'] + electrode_cols].copy()
plt.figure(figsize=(12,4))
plt.plot(df_eeg_temp['datetime'], df_eeg_temp['EEG.AF3'], label='Raw AF3')
plt.title('EEG Raw - AF3')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(dir_out, 'eeg_raw_af3.png'))
plt.close()

# 5) Imputación de ceros
df_eeg_imputed = eeg.copy()
for ch in electrode_cols:
    df_eeg_imputed[ch] = df_eeg_imputed[ch].replace(0.0, np.nan).interpolate(method='linear', limit_direction='both')

# 6) Remoción de offset
df_eeg_base = df_eeg_imputed.copy()
for ch in electrode_cols:
    df_eeg_base[ch] -= df_eeg_base[ch].mean()

plt.figure(figsize=(12,4))
plt.plot(df_eeg_base['datetime'], df_eeg_base['EEG.AF3'], label='Offset Removido AF3')
plt.title('EEG Offset Removed - AF3')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(dir_out, 'eeg_offset_removed.png'))
plt.close()

# 7) Filtrado Notch y pasa-bajo
fs = 2048
b50,a50 = iirnotch(50,30,fs)
b60,a60 = iirnotch(60,30,fs)
nyq = fs/2
b_low,a_low = butter(4, 64/nyq, btype='low')
df_eeg_filtered = df_eeg_base.copy()
for ch in electrode_cols:
    sig = df_eeg_filtered[ch].values
    sig = filtfilt(b50,a50,sig)
    sig = filtfilt(b60,a60,sig)
    sig = filtfilt(b_low,a_low,sig)
    df_eeg_filtered[ch] = sig

plt.figure(figsize=(12,4))
plt.plot(df_eeg_filtered['datetime'], df_eeg_filtered['EEG.AF3'], label='Filtered AF3')
plt.title('EEG Filtered - AF3')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(dir_out, 'eeg_filtered_af3.png'))
plt.close()

# 8) Downsampling a 256 Hz
down = int(fs/256)
df_eeg_down = df_eeg_filtered.iloc[::down].copy()
for ch in electrode_cols:
    df_eeg_down[ch] = decimate(df_eeg_filtered[ch].values, down, zero_phase=True)

plt.figure(figsize=(12,4))
plt.plot(df_eeg_down['datetime'], df_eeg_down['EEG.AF3'], label='Downsampled AF3')
plt.title('EEG Downsampled 256Hz - AF3')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(dir_out, 'eeg_downsampled_af3.png'))
plt.close()

# 9) Preparar ICA
X = df_eeg_filtered[electrode_cols].values
ica = FastICA(n_components=len(electrode_cols), random_state=42)
sources = ica.fit_transform(X)
kurt_vals = kurtosis(sources, fisher=True)

# Guardar kurtosis
df_kurt = pd.DataFrame({
    'component': range(len(kurt_vals)),
    'kurtosis': kurt_vals
})
df_kurt.to_csv(os.path.join(dir_out, 'ica_kurtosis.csv'), index=False)

# Plot kurtosis
plt.figure(figsize=(8,4))
plt.bar(df_kurt['component'], df_kurt['kurtosis'])
plt.title('ICA Kurtosis per Component')
plt.xlabel('Componente')
plt.ylabel('Kurtosis')
plt.tight_layout()
plt.savefig(os.path.join(dir_out, 'ica_kurtosis.png'))
plt.close()

# Identificar artefactos, reconstruir señal limpia
th = 312.0
art_idx = np.where(np.abs(kurt_vals)>th)[0]
sources_clean = sources.copy()
sources_clean[:,art_idx] = 0
reconstructed = ica.inverse_transform(sources_clean)

df_eeg_ica = df_eeg_filtered.copy()
for i,ch in enumerate(electrode_cols):
    df_eeg_ica[ch] = reconstructed[:,i]

# 10) Preparar MNE RawArray y aplicar ICA limpia
# Lista de canales EEG
electrode_cols = [
    'EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5',
    'EEG.T7', 'EEG.P7', 'EEG.O1', 'EEG.O2',
    'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4',
    'EEG.F8', 'EEG.AF4'
]

# Lista de sujetos a procesar (se asume que la columna 'subject' existe)
subjects = df_eeg_filtered['subject'].unique()
print("Número de sujetos únicos en df_eeg_down:", len(subjects))
print("Sujetos encontrados:", subjects)

# Lista para almacenar los DataFrames procesados por sujeto
list_dfs = []
# Procesamiento por sujeto
for subj in subjects:
    print(f"\nProcesando sujeto: {subj}")
    # Extraer los datos correspondientes a este sujeto
    df_subj = df_eeg_filtered[df_eeg_filtered['subject'] == subj].copy()
    
    # Verificar la longitud y rango de tiempo para este sujeto
    print(f"  Número de muestras: {len(df_subj)}")
    print(f"  Rango de tiempo: {df_subj['datetime'].min()} a {df_subj['datetime'].max()}")
    
    # Extraer la matriz de datos EEG (de los 14 canales) de este sujeto
    # La forma debe ser (n_samples, n_channels)
    X_subj = df_subj[electrode_cols].values
    
    # Transponer para que RawArray reciba una matriz de forma (n_channels, n_samples)
    X_subj_T = X_subj.T
    
    # Definir la frecuencia de muestreo (por ejemplo, 256 Hz, que es la de df_eeg_down)
    sfreq = 256
    
    # Crear la información de los canales
    info = mne.create_info(ch_names=electrode_cols, sfreq=sfreq, ch_types=['eeg'] * len(electrode_cols))
    
    # Crear el objeto RawArray a partir de los datos EEG
    raw_subj = mne.io.RawArray(X_subj_T, info)

    
    # Renombrar los canales para que coincidan con el montaje estándar (quitando "EEG.")
    mapping = {ch: ch.replace("EEG.", "") for ch in raw_subj.info['ch_names']}
    raw_subj.rename_channels(mapping)
    
    # Asignar un montage estándar para que se agreguen las posiciones de los electrodos
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_subj.set_montage(montage, on_missing='warn')
    
    # Muestra la localización de los sensores en el montaje asignado (standard_1020)
    raw_subj.plot_sensors(show_names=True)
    
    

    # Aplicar ICA con MNE para este sujeto

    # Aplicar un filtro pasa-alto con límite inferior de 1 Hz
    raw_subj.filter(l_freq=1.0, h_freq=None, fir_design='firwin')

    # Ahora ajustamos el ICA
    ica = ICA(n_components=len(raw_subj.info['ch_names']), random_state=42, max_iter='auto')
    ica.fit(raw_subj)
    print("ICA ajustada correctamente tras aplicar el filtro high-pass.")

    # Visualizar las componentes para inspección manual
    ica.plot_components(show=True)

    # Muestra cómo evoluciona en el tiempo cada componente independiente
    ica.plot_sources(raw_subj, show=True, start=0, stop=60)
    
    # Inspeccionar en detalle las propiedades (topografía, espectro, forma de onda) de la componente 0
    ica.plot_properties(raw_subj, picks=[0], psd_args={'fmax':50})
    
    # Obtener las fuentes (componentes) como un array (forma: n_components x n_samples)
    sources = ica.get_sources(raw_subj).get_data()
    kurt_vals = kurtosis(sources, axis=1, fisher=True)
    print("  Kurtosis de cada componente:")
    for i, k in enumerate(kurt_vals):
        print(f"    Componente {i}: {k:.2f}")
    
    # Definir un umbral para eliminar componentes artefactuosas
    threshold = 312.0
    artifact_indices = np.where(np.abs(kurt_vals) > threshold)[0]
    print("  Componentes identificadas como artefacto (|kurtosis| > 312):", artifact_indices)
    
    # Excluir las componentes identificadas (puedes ajustarlo manualmente si lo deseas)
    ica.exclude = list(artifact_indices)
    print("  Componentes excluidas:", ica.exclude)
    
    # Aplicar ICA para remover los artefactos y obtener la señal limpia
    raw_subj_clean = ica.apply(raw_subj.copy())
    print("  Señal limpia obtenida para el sujeto.")
    
    
    # Convertir la señal limpia a DataFrame y conservar el timestamp original

    # Extraer los datos limpios: shape (n_channels, n_samples)
    eeg_clean = raw_subj_clean.get_data()
    # Transponer para tener (n_samples, n_channels)
    eeg_clean = eeg_clean.T
    
    # Crear un DataFrame con las columnas correspondientes a los electrodos (usamos los nombres ya renombrados)
    # Nota: raw_subj_clean.info['ch_names'] ya son, por ejemplo, "AF3", "F7", etc.
    df_subj_clean = pd.DataFrame(eeg_clean, columns=raw_subj_clean.info['ch_names'])
    
    # Importante: asignar los timestamps originales del sujeto, que están en df_subj['datetime']
    # Se asume que el número de muestras es el mismo
    df_subj_clean['datetime'] = df_subj['datetime'].values  # preservamos los timestamps originales
    df_subj_clean['subject'] = subj  # añadir el identificador del sujeto
    df_subj_clean['theoretical_difficulty'] = df_subj['theoretical_difficulty'].values
    
    list_dfs.append(df_subj_clean)
# Concatenar los DataFrames de todos los sujetos
df_eeg_ica_final = pd.concat(list_dfs, axis=0).reset_index(drop=True)
print("\nForma final del DataFrame con datos limpios:", df_eeg_ica_final.shape)
print("Rango de tiempo final:", df_eeg_ica_final['datetime'].min(), "a", df_eeg_ica_final['datetime'].max())
# Visualización de ejemplo: canal AF3 para el primer sujeto
# (Nota: después de renombrar, el canal se llama "AF3" y no "EEG.AF3")
subject_example = df_eeg_ica_final[df_eeg_ica_final['subject'] == subjects[0]]
plt.figure(figsize=(12, 4))
plt.plot(subject_example['datetime'], subject_example['AF3'], label="AF3 Limpio")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.title("Señal limpia tras ICA (MNE) - Canal AF3, sujeto " + str(subjects[0]))
plt.legend()
plt.show()
# Visualización de ejemplo: canal AF3 para el segundo sujeto
# (Nota: después de renombrar, el canal se llama "AF3" y no "EEG.AF3")
subject_example = df_eeg_ica_final[df_eeg_ica_final['subject'] == subjects[1]]
plt.figure(figsize=(12, 4))
plt.plot(subject_example['datetime'], subject_example['AF3'], label="AF3 Limpio")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.title("Señal limpia tras ICA (MNE) - Canal AF3, sujeto " + str(subjects[1]))
plt.legend()
plt.show()

# 11) Guardar df_eeg_ica_final como CSV
df_eeg_ica_final.to_csv(os.path.join(dir_out, 'df_eeg_ica_final.csv'), index=False)

print('Preprocesamiento completado. Resultados en:', dir_out)
