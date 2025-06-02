import os
import glob
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
from scipy.signal import iirnotch, butter, filtfilt, decimate, welch
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Funciones para leer cada tipo de archivo
def leer_parquet(ruta):
    return pd.read_parquet(ruta)

def leer_csv(ruta):
    return pd.read_csv(ruta)

def leer_json(ruta):
    with open(ruta, 'r', encoding='utf-8') as archivo:
        return json.load(archivo)

# Directorio base de la base de datos
base_dir = "/Users/danillopis/Desktop/TFG/UAB/workload_dataset"

# --- Lectura de datos: Flight Simulation ---
flight_sim_dir = os.path.join(base_dir, "data_flight_simulator")

# ECG: ecg_hr.parquet y ecg_ibi.parquet
ecg_hr_path = os.path.join(flight_sim_dir, "ecg", "ecg_hr.parquet")
ecg_ibi_path = os.path.join(flight_sim_dir, "ecg", "ecg_ibi.parquet")
df_ecg_hr = leer_parquet(ecg_hr_path)
df_ecg_ibi = leer_parquet(ecg_ibi_path)

# EEG: eeg.parquet
eeg_path = os.path.join(flight_sim_dir, "eeg", "eeg.parquet")
df_eeg = leer_parquet(eeg_path)

# Perceived difficulty (archivos JSON)
json_dir = os.path.join(flight_sim_dir, "perceived_difficulty")
json_files = ["flight_1.json", "flight_2_4.json", "flight_3_5.json"]
perceived_difficulty = {}
for jf in json_files:
    ruta_json = os.path.join(json_dir, jf)
    perceived_difficulty[jf] = leer_json(ruta_json)

# --- Lectura de datos: Heat-the-Chair ---
htc_dir = os.path.join(base_dir, "data_heat_the_chair")

# ECG: ecg.parquet
ecg_htc_path = os.path.join(htc_dir, "ecg", "ecg.parquet")
df_ecg_htc = leer_parquet(ecg_htc_path)

# EEG: eeg.parquet
eeg_htc_path = os.path.join(htc_dir, "eeg", "eeg.parquet")
df_eeg_htc = leer_parquet(eeg_htc_path)

# Game performance (archivos CSV)
game_perf_dir = os.path.join(htc_dir, "game_performance")
csv_files = glob.glob(os.path.join(game_perf_dir, "*.csv"))
game_performance = {}
for archivo in csv_files:
    nombre = os.path.basename(archivo)
    game_performance[nombre] = leer_csv(archivo)

# Subjective performance: tlx_answers.parquet
subjective_htc_path = os.path.join(htc_dir, "subjective_performance", "tlx_answers.parquet")
df_subjective_htc = leer_parquet(subjective_htc_path)

# --- Lectura de datos: N-back test ---
nback_dir = os.path.join(base_dir, "data_n_back_test")

# ECG: ecg_br.parquet, ecg_hr.parquet y ecg_ibi.parquet
ecg_br_path = os.path.join(nback_dir, "ecg", "ecg_br.parquet")
ecg_hr_nback_path = os.path.join(nback_dir, "ecg", "ecg_hr.parquet")
ecg_ibi_nback_path = os.path.join(nback_dir, "ecg", "ecg_ibi.parquet")
df_ecg_br = leer_parquet(ecg_br_path)
df_ecg_hr_nback = leer_parquet(ecg_hr_nback_path)
df_ecg_ibi_nback = leer_parquet(ecg_ibi_nback_path)

# EEG: eeg.parquet
eeg_nback_path = os.path.join(nback_dir, "eeg", "eeg.parquet")
df_eeg_nback = leer_parquet(eeg_nback_path)

# Game performance: game_scores.parquet
game_scores_path = os.path.join(nback_dir, "game_performance", "game_scores.parquet")
df_game_scores = leer_parquet(game_scores_path)

# Subjective performance: tlx_answers.parquet
subjective_nback_path = os.path.join(nback_dir, "subjective_performance", "tlx_answers.parquet")
df_subjective_nback = leer_parquet(subjective_nback_path)

# Muestra de datos leídos
print("Flight Simulation - EEG:")
print(df_eeg.head(), "\n")

print("Flight Simulation - ECG HR:")
print(df_ecg_hr.head(), "\n")

print("Heat-the-Chair - ECG:")
print(df_ecg_htc.head(), "\n")

print("Heat-the-Chair - EEG:")
print(df_eeg_htc.head(), "\n")

print("N-back Test - EEG:")
print(df_eeg_nback.head(), "\n")

print("N-back Test - ECG:")
print(df_ecg_ibi_nback.head(), "\n")


# Exportamos cada pieza de datos a CSVs

output_dir = os.path.join(base_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

# Flight Simulation
df_ecg_hr.to_csv(os.path.join(output_dir, "df_ecg_hr.csv"), index=False)
df_ecg_ibi.to_csv(os.path.join(output_dir, "df_ecg_ibi.csv"), index=False)
df_eeg.to_csv(os.path.join(output_dir, "df_eeg.csv"), index=False)

for jf, data in perceived_difficulty.items():
    pd.DataFrame(data).to_csv(
        os.path.join(output_dir, jf.replace(".json", ".csv")),
        index=False
    )

# Heat-the-Chair
df_ecg_htc.to_csv(os.path.join(output_dir, "df_ecg_htc.csv"), index=False)
df_eeg_htc.to_csv(os.path.join(output_dir, "df_eeg_htc.csv"), index=False)
df_subjective_htc.to_csv(os.path.join(output_dir, "df_subjective_htc.csv"), index=False)

for nombre, df in game_performance.items():
    df.to_csv(
        os.path.join(output_dir, nombre.replace(".csv", ".csv")),
        index=False
    )

# N-back Test
df_ecg_br.to_csv(os.path.join(output_dir, "df_ecg_br.csv"), index=False)
df_ecg_hr_nback.to_csv(os.path.join(output_dir, "df_ecg_hr_nback.csv"), index=False)
df_ecg_ibi_nback.to_csv(os.path.join(output_dir, "df_ecg_ibi_nback.csv"), index=False)
df_eeg_nback.to_csv(os.path.join(output_dir, "df_eeg_nback.csv"), index=False)
df_game_scores.to_csv(os.path.join(output_dir, "df_game_scores.csv"), index=False)
df_subjective_nback.to_csv(os.path.join(output_dir, "df_subjective_nback.csv"), index=False)

print("Todos los CSV se han guardado en:", output_dir)