import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch

# Rutas
base_dir = "/Users/danillopis/Desktop/TFG/UAB/workload_dataset"
input_dir = os.path.join(base_dir, "outputs_preproc")
output_dir = os.path.join(base_dir, "outputs_features")
os.makedirs(output_dir, exist_ok=True)

# Carga del DataFrame preprocesado con ICA
df_eeg_ica = pd.read_csv(os.path.join(input_dir, "df_eeg_ica_final.csv"))
# Asegurar formato datetime
df_eeg_ica['datetime'] = pd.to_datetime(df_eeg_ica['datetime'])

# Funciones de extracción de características
def compute_time_features(signal):
    features = {}
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['var'] = np.var(signal)
    features['skew'] = skew(signal)
    features['kurtosis'] = kurtosis(signal, fisher=True)
    features['ptp'] = np.ptp(signal)
    zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
    features['zero_crossings'] = len(zero_crossings)
    return features

# Parámetros de extracción
fs = 256  # frecuencia de muestreo
window_duration = 1.0  # segundos
window_size = int(fs * window_duration)
bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}
electrode_cols = [
    'AF3','F7','F3','FC5',
    'T7','P7','O1','O2',
    'P8','T8','FC6','F4',
    'F8','AF4'
]

# Construcción de características por ventana
features_list = []
num_samples = len(df_eeg_ica)
for start in range(0, num_samples - window_size + 1, window_size):
    window = df_eeg_ica.iloc[start:start + window_size]
    wf = {
        'start_time': window['datetime'].iloc[0],
        'subject': window['subject'].iloc[0],
        'theoretical_difficulty': window['theoretical_difficulty'].mode()[0]
    }
    # PSD de la ventana
    for ch in electrode_cols:
        sig = window[ch].values
        # 1) Características de tiempo
        t_feats = compute_time_features(sig)
        # 2) PSD y potencia por banda
        freqs, psd = welch(sig, fs=fs, nperseg=window_size, nfft=512)
        dfreq = freqs[1] - freqs[0]
        f_feats = {}
        total_bp = 0.0
        for bn, br in bands.items():
            bp = np.sum(psd[(freqs >= br[0]) & (freqs <= br[1])]) * dfreq
            f_feats[f'{bn}_power'] = bp
            total_bp += bp
        for bn in bands:
            f_feats[f'{bn}_power_rel'] = f_feats[f'{bn}_power'] / total_bp if total_bp > 0 else 0.0
        # Theta/alpha ratio
        a_bp = f_feats['alpha_power']
        f_feats['theta_alpha_ratio'] = f_feats['theta_power'] / a_bp if a_bp > 0 else 0.0
        # Entropía espectral
        psd_norm = psd / (psd.sum() + 1e-12)
        f_feats['spectral_entropy'] = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
        # Integrar al dict
        for k, v in {**t_feats, **f_feats}.items():
            wf[f'{ch}_{k}'] = v
    # Asimetría frontal
    aR = wf['AF4_alpha_power']; aL = wf['AF3_alpha_power']
    wf['frontal_alpha_asym'] = np.log(aR + 1e-10) - np.log(aL + 1e-10)
    features_list.append(wf)

# DataFrame de características
df_features = pd.DataFrame(features_list)
# Mapear etiquetas de dificultad
df_features['theoretical_difficulty'] = df_features['theoretical_difficulty'].replace(-1, 0)
group_map = {0:0,1:1,2:2,3:2,4:2}
df_features['theoretical_difficulty'] = df_features['theoretical_difficulty'].map(group_map).astype(int)

# Guardar resultados
features_csv = os.path.join(output_dir, 'features.csv')
df_features.to_csv(features_csv, index=False, encoding='utf-8')
print("Extracción de características completada. CSV guardado en:", features_csv)
