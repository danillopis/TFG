# eda.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools

# Directorio base y carpetas de entrada/salida
base_dir = "/Users/danillopis/Desktop/TFG/UAB/workload_dataset"
input_dir = os.path.join(base_dir, "outputs")
output_dir = os.path.join(base_dir, "outputs_eda")
os.makedirs(output_dir, exist_ok=True)

# Carga de los CSV generados por data_load.py
df_ecg_hr = pd.read_csv(os.path.join(input_dir, "df_ecg_hr.csv"))
df_ecg_ibi = pd.read_csv(os.path.join(input_dir, "df_ecg_ibi.csv"))
df_eeg    = pd.read_csv(os.path.join(input_dir, "df_eeg.csv"))

print("Forma del DataFrame ECG (Heart Rate):", df_ecg_hr.shape)
print("Forma del DataFrame ECG (IBI):", df_ecg_ibi.shape)
print("Forma del DataFrame EEG:", df_eeg.shape)
print("\nPrimeras filas de ECG Heart Rate:")
df_ecg_hr.head()
print("\nPrimeras filas de ECG IBI:")
df_ecg_ibi.head()
print("\nPrimeras filas de EEG:")
df_eeg.head()
print("EEG Info:")
print(df_eeg.info())
# Seleccionar columnas por tipo
object_columns = df_eeg.select_dtypes(include=['object']).columns.tolist()
int16_columns = df_eeg.select_dtypes(include=['int16']).columns.tolist()
int32_columns = df_eeg.select_dtypes(include=['int32']).columns.tolist()
int64_columns = df_eeg.select_dtypes(include=['int64']).columns.tolist()
int8_columns  = df_eeg.select_dtypes(include=['int8']).columns.tolist()

# Imprimir resultados
print("\nColumnas de tipo object:")
print(object_columns)

print("\nColumnas de tipo int16 (cantidad: {}):".format(len(int16_columns)))
print(int16_columns)

print("\nColumnas de tipo int32 (cantidad: {}):".format(len(int32_columns)))
print(int32_columns)

print("\nColumnas de tipo int64 (cantidad: {}):".format(len(int64_columns)))
print(int64_columns)

print("\nColumnas de tipo int8 (cantidad: {}):".format(len(int8_columns)))
print(int8_columns)
print("\nECG(HR) Info:")
print(df_ecg_hr.info())
print("\nECG(IBI) Info:")
print(df_ecg_ibi.info())
print("\nEstadísticas descriptivas de ECG Heart Rate:")
print(df_ecg_hr.describe())

print("\nEstadísticas descriptivas de ECG IBI:")
print(df_ecg_ibi.describe())

print("\nEstadísticas descriptivas de EEG:")
print(df_eeg.describe())
# Aseguramos que la columna 'datetime' esté en formato datetime
df_eeg['datetime'] = pd.to_datetime(df_eeg['datetime'])
df_ecg_hr['datetime'] = pd.to_datetime(df_ecg_hr['datetime'])
df_ecg_ibi['datetime'] = pd.to_datetime(df_ecg_ibi['datetime'])

print("----- Sincronización Temporal -----")
print("EEG - Mínimo datetime:", df_eeg['datetime'].min())
print("EEG - Máximo datetime:", df_eeg['datetime'].max())
print("ECG (HR) - Mínimo datetime:", df_ecg_hr['datetime'].min())
print("ECG (HR) - Máximo datetime:", df_ecg_hr['datetime'].max())
print("ECG (IBI) - Mínimo datetime:", df_ecg_ibi['datetime'].min())
print("ECG (IBI) - Máximo datetime:", df_ecg_ibi['datetime'].max())
df_subj1 = df_eeg[df_eeg['subject'] == 1]
print("Rango de tiempo sujeto 1:", df_subj1['datetime'].min(), df_subj1['datetime'].max())
print("Número de muestras:", len(df_subj1))
df_subj2 = df_eeg[df_eeg['subject'] == 2]
print("Rango de tiempo sujeto 2:", df_subj2['datetime'].min(), df_subj2['datetime'].max())
print("Número de muestras:", len(df_subj2))
df_subj1 = df_ecg_hr[df_ecg_hr['subject'] == 1]
print("Rango de tiempo sujeto 1:", df_subj1['datetime'].min(), df_subj1['datetime'].max())
print("Número de muestras:", len(df_subj1))
df_subj1 = df_ecg_hr[df_ecg_hr['subject'] == 2]
print("Rango de tiempo sujeto 2:", df_subj1['datetime'].min(), df_subj1['datetime'].max())
print("Número de muestras:", len(df_subj1))
df_subj1= df_ecg_ibi[df_ecg_ibi['subject'] == 1]
print("Rango de tiempo sujeto 1:", df_subj1['datetime'].min(), df_subj1['datetime'].max())
print("Número de muestras:", len(df_subj1))
df_subj2 = df_ecg_ibi[df_ecg_ibi['subject'] == 2]
print("Rango de tiempo sujeto 2:", df_subj2['datetime'].min(), df_subj2['datetime'].max())
print("Número de muestras:", len(df_subj2))
region_upper_left = ["EEG.F7", "EEG.AF3", "EEG.F3", "EEG.FC5"]
region_upper_right = ["EEG.F8", "EEG.AF4", "EEG.F4", "EEG.FC6"]
region_lower_left = ["EEG.T7", "EEG.P7", "EEG.O1"]
region_lower_right = ["EEG.T8", "EEG.P8", "EEG.O2"]
# En este ejemplo graficamos solo las primeras 1000 muestras para no saturar la vista.
n_muestras = 1000

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# --- Región Superior Izquierda ---
df_temp = df_eeg[region_upper_left].iloc[:n_muestras]
for ch in region_upper_left:
    axes[0, 0].plot(df_temp.index, df_temp[ch], label=ch)
axes[0, 0].set_title("EEG - Región Superior Izquierda")
axes[0, 0].set_xlabel("Muestra")
axes[0, 0].set_ylabel("Amplitud (µV)")
axes[0, 0].legend(loc="best")

# --- Región Superior Derecha ---
df_temp = df_eeg[region_upper_right].iloc[:n_muestras]
for ch in region_upper_right:
    axes[0, 1].plot(df_temp.index, df_temp[ch], label=ch)
axes[0, 1].set_title("EEG - Región Superior Derecha")
axes[0, 1].set_xlabel("Muestra")
axes[0, 1].set_ylabel("Amplitud (µV)")
axes[0, 1].legend(loc="best")

# --- Región Inferior Izquierda ---
df_temp = df_eeg[region_lower_left].iloc[:n_muestras]
for ch in region_lower_left:
    axes[1, 0].plot(df_temp.index, df_temp[ch], label=ch)
axes[1, 0].set_title("EEG - Región Inferior Izquierda")
axes[1, 0].set_xlabel("Muestra")
axes[1, 0].set_ylabel("Amplitud (µV)")
axes[1, 0].legend(loc="best")

# --- Región Inferior Derecha ---
df_temp = df_eeg[region_lower_right].iloc[:n_muestras]
for ch in region_lower_right:
    axes[1, 1].plot(df_temp.index, df_temp[ch], label=ch)
axes[1, 1].set_title("EEG - Región Inferior Derecha")
axes[1, 1].set_xlabel("Muestra")
axes[1, 1].set_ylabel("Amplitud (µV)")
axes[1, 1].legend(loc="best")

plt.tight_layout()
plt.show()
# Calculamos la media de cada región por fila (tiempo)
df_eeg['mean_upper_left'] = df_eeg[region_upper_left].mean(axis=1)
df_eeg['mean_upper_right'] = df_eeg[region_upper_right].mean(axis=1)
df_eeg['mean_lower_left'] = df_eeg[region_lower_left].mean(axis=1)
df_eeg['mean_lower_right'] = df_eeg[region_lower_right].mean(axis=1)
# A modo de ejemplo, un boxplot agrupado por 'perceived_difficulty'
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

sns.boxplot(x='perceived_difficulty', y='mean_upper_left', data=df_eeg, ax=axes[0,0])
axes[0,0].set_title("Región Superior Izquierda vs. Dificultad")
axes[0,0].set_xlabel("Dificultad Percibida")
axes[0,0].set_ylabel("Amplitud Media (µV)")

sns.boxplot(x='perceived_difficulty', y='mean_upper_right', data=df_eeg, ax=axes[0,1])
axes[0,1].set_title("Región Superior Derecha vs. Dificultad")
axes[0,1].set_xlabel("Dificultad Percibida")
axes[0,1].set_ylabel("Amplitud Media (µV)")

sns.boxplot(x='perceived_difficulty', y='mean_lower_left', data=df_eeg, ax=axes[1,0])
axes[1,0].set_title("Región Inferior Izquierda vs. Dificultad")
axes[1,0].set_xlabel("Dificultad Percibida")
axes[1,0].set_ylabel("Amplitud Media (µV)")

sns.boxplot(x='perceived_difficulty', y='mean_lower_right', data=df_eeg, ax=axes[1,1])
axes[1,1].set_title("Región Inferior Derecha vs. Dificultad")
axes[1,1].set_xlabel("Dificultad Percibida")
axes[1,1].set_ylabel("Amplitud Media (µV)")

plt.tight_layout()
plt.show()
# Distribución de HR y IBI
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.histplot(df_ecg_hr['hr'], kde=True, color='blue')
plt.title("Distribución de HR (Frecuencia Cardíaca)")
plt.xlabel("HR (bpm)")

plt.subplot(1, 2, 2)
sns.histplot(df_ecg_ibi['rr_int'], kde=True, color='orange')
plt.title("Distribución de IBI (RR Interval)")
plt.xlabel("RR Interval (ms)")

plt.tight_layout()
plt.show()
# Distribución de HR por fase de vuelo
plt.figure(figsize=(10, 6))
sns.boxplot(x='phase', y='hr', data=df_ecg_hr, palette='Set2')
plt.title("Distribución de HR por Fase de Vuelo")
plt.xlabel("Fase de Vuelo")
plt.ylabel("HR (bpm)")
plt.show()
# Distribución de HR por dificultad percibida
plt.figure(figsize=(10, 6))
sns.boxplot(x='perceived_difficulty', y='hr', data=df_ecg_hr, palette='coolwarm')
plt.title("Distribución de HR por Dificultad Percibida")
plt.xlabel("Dificultad Percibida")
plt.ylabel("HR (bpm)")
plt.show()
# Distribución de HR por dificultad percibida y dado un vuelo y uno de los pilotos
flight = 1
subject = 2

df_temp_hr = df_ecg_hr[(df_ecg_hr['flight'] == flight) & (df_ecg_hr['subject'] == subject)] # Filtrar por vuelo y piloto

if not df_temp_hr.empty:
    df_temp_hr = df_temp_hr.sort_values(by='datetime')

    plt.figure(figsize=(12, 5))

    # Graficar HR
    plt.plot(df_temp_hr['datetime'], df_temp_hr['hr'], label='HR (bpm)')

    # Escalar perceived_difficulty para que se vea en la misma gráfica
    plt.scatter(df_temp_hr['datetime'], df_temp_hr['perceived_difficulty'] * 10, # simplemente por escalar y que se pueda ver en la misma gráfica
                color='red', alpha=0.5, label='perceived_difficulty (x10)')

    plt.title(f"Evolución de HR y Dificultad Percibida (flight={flight}, subject={subject})")
    plt.xlabel("Tiempo")
    plt.ylabel("Frecuencia Cardíaca (bpm)")
    plt.legend()
    plt.show()
else:
    print(f"No hay datos para flight={flight} y subject={subject}")
# Cálculo de la correlación entre HR y dificultad percibida

# Cálculo de la correlación (Pearson) a nivel global
corr_value = df_ecg_hr[['hr', 'perceived_difficulty']].corr().iloc[0,1]
print(f"Correlación global HR vs Dificultad Percibida: {corr_value:.2f}")

# También podemos agrupar por fase o vuelo y ver la correlación en cada caso MIRAAAAAAAAAR NO DA
grouped_corr = df_ecg_hr.groupby(['phase'])[['hr','perceived_difficulty']].corr().iloc[0::2,-1].reset_index()
print("\nCorrelación por fase:")
print(grouped_corr)
# Distribución de IBI por fase de vuelo
plt.figure(figsize=(10, 6))
sns.boxplot(x='phase', y='rr_int', data=df_ecg_ibi, palette='Set3')
plt.title("Distribución de IBI (RR Interval) por Fase de Vuelo")
plt.xlabel("Fase de Vuelo")
plt.ylabel("RR Interval (ms)")
plt.show()
# Distribución de IBI por dificultad percibida
plt.figure(figsize=(10, 6))
sns.boxplot(x='perceived_difficulty', y='rr_int', data=df_ecg_ibi, palette='viridis')
plt.title("Distribución de IBI por Dificultad Percibida")
plt.xlabel("Dificultad Percibida")
plt.ylabel("RR Interval (ms)")
plt.show()
dataframes = {
    "EEG": df_eeg,
    "ECG (HR)": df_ecg_hr,
    "ECG (IBI)": df_ecg_ibi
}

for name, df in dataframes.items():
    print(f"=== Análisis de nulos en {name} ===")
    
    # Filas con TODOS los valores nulos
    all_null_rows = df.isnull().all(axis=1).sum()
    print(f"Filas con todos los valores nulos: {all_null_rows}")
    
    # Lista de columnas que tienen al menos un valor nulo
    columnas_con_nulos = [col for col in df.columns if df[col].isnull().any()]
    
    if columnas_con_nulos:
        print("Columnas con nulos:")
        for col in columnas_con_nulos:
            nulos = df[col].isnull().sum()
            print(f" - {col}: {nulos} nulos")
    else:
        print("No se encontraron columnas con valores nulos.")
    
    print("-" * 40)
# Definimos los conjuntos de valores permitidos
allowed_subject = {1, 2}
allowed_flight = {1, 2, 3, 4, 5}
allowed_phase = {"baseline", "flight"}
# Para perceived_difficulty, permitimos valores enteros entre -1 y 4 (incluidos)
allowed_perceived = set(range(-1, 4))    # Esto da {-1, 0, 1, 2, 3}
# Para theoretical_difficulty, permitimos enteros entre -1 y 3 (incluidos)
allowed_theoretical = set(range(-1, 5))    # Esto da {-1, 0, 1, 2, 3, 4}
# Función para verificar una columna y mostrar las filas que no cumplen
def verificar_columna(df, columna, allowed_values):
    # Seleccionamos las filas donde el valor no está en el conjunto permitido
    invalid = df[~df[columna].isin(allowed_values)]
    return invalid
# Iteramos sobre cada dataset y revisamos cada condición
for ds_name, df in dataframes.items():
    print(f"=== Dataset: {ds_name} ===")
    
    # Verificar 'subject'
    invalid_subject = verificar_columna(df, 'subject', allowed_subject)
    if not invalid_subject.empty:
        print("Filas con 'subject' inválido:")
        print(invalid_subject[['subject']])
    else:
        print("Todos los valores de 'subject' son válidos.")
    
    # Verificar 'flight'
    invalid_flight = verificar_columna(df, 'flight', allowed_flight)
    if not invalid_flight.empty:
        print("Filas con 'flight' inválido:")
        print(invalid_flight[['flight']])
    else:
        print("Todos los valores de 'flight' son válidos.")
    
    # Verificar 'phase'
    invalid_phase = verificar_columna(df, 'phase', allowed_phase)
    if not invalid_phase.empty:
        print("Filas con 'phase' inválido:")
        print(invalid_phase[['phase']])
    else:
        print("Todos los valores de 'phase' son válidos.")
    
    # Verificar 'perceived_difficulty'
    invalid_perceived = verificar_columna(df, 'perceived_difficulty', allowed_perceived)
    if not invalid_perceived.empty:
        print("Filas con 'perceived_difficulty' inválido:")
        print(invalid_perceived[['perceived_difficulty']])
    else:
        print("Todos los valores de 'perceived_difficulty' son válidos.")
    
    # Verificar 'theoretical_difficulty'
    invalid_theoretical = verificar_columna(df, 'theoretical_difficulty', allowed_theoretical)
    if not invalid_theoretical.empty:
        print("Filas con 'theoretical_difficulty' inválido:")
        print(invalid_theoretical[['theoretical_difficulty']])
    else:
        print("Todos los valores de 'theoretical_difficulty' son válidos.")
    
    print("-" * 60)
# Valores permitidos para subject y flight 
allowed_subjects = {1, 2}
allowed_flights = {1, 2, 3, 4, 5}

# Generar todas las combinaciones esperadas
expected_combinations = pd.DataFrame(
    list(itertools.product(allowed_subjects, allowed_flights)),
    columns=['subject', 'flight']
)

def find_missing_combinations(df, df_name, expected_df):
    """
    Encuentra las combinaciones de ['subject','flight'] que están en expected_df pero no en el dataset df.
    """
    # Extraer combinaciones únicas presentes en el dataset
    df_combinations = df[['subject', 'flight']].drop_duplicates()
    # Merge para identificar combinaciones faltantes
    merged = expected_df.merge(df_combinations, on=['subject', 'flight'], how='left', indicator=True)
    missing = merged[merged['_merge'] == 'left_only'][['subject', 'flight']]
    
    if missing.empty:
        print(f"Todas las combinaciones de subject y flight están presentes en {df_name}.")
    else:
        print(f"\nFaltan las siguientes combinaciones en {df_name}:")
        print(missing)
    
    return missing

# Buscar las combinaciones faltantes en cada dataset
missing_eeg = find_missing_combinations(df_eeg, "df_eeg", expected_combinations)
missing_ecg_hr = find_missing_combinations(df_ecg_hr, "df_ecg_hr", expected_combinations)
missing_ecg_ibi = find_missing_combinations(df_ecg_ibi, "df_ecg_ibi", expected_combinations)

def verify_missing_combinations(df, missing_df, df_name):
    """
    Para cada combinación faltante reportada, verifica que realmente no existan registros en el dataset df.
    """
    for idx, row in missing_df.iterrows():
        subj = row['subject']
        flight = row['flight']
        subset = df[(df['subject'] == subj) & (df['flight'] == flight)]
        if subset.empty:
            print(f"Verificación: La combinación subject={subj}, flight={flight} NO existe en {df_name} (correcto).")
        else:
            print(f"Verificación: ¡Atención! La combinación subject={subj}, flight={flight} SÍ existe en {df_name}:")
            print(subset)

# Verificar en cada dataset
print("\n--- Verificación en df_eeg ---")
verify_missing_combinations(df_eeg, missing_eeg, "df_eeg")

print("\n--- Verificación en df_ecg_hr ---")
verify_missing_combinations(df_ecg_hr, missing_ecg_hr, "df_ecg_hr")

print("\n--- Verificación en df_ecg_ibi ---")
verify_missing_combinations(df_ecg_ibi, missing_ecg_ibi, "df_ecg_ibi")
if "role" in df_eeg.columns:
    unique_roles = df_eeg["role"].unique()
    print("Valores únicos en 'role' en df_eeg:")
    print(unique_roles)
else:
    print("La columna 'role' no existe en df_eeg.")
for name, df in dataframes.items():
    if "phase" in df.columns:
        unique_phases = df["phase"].unique()
        print(f"Valores únicos en 'phase' en {name}:")
        print(unique_phases)
    else:
        print(f"La columna 'phase' no existe en el dataset {name}.")
# Definir las columnas que NO deseamos incluir en la matriz de correlación
exclusion_keywords = ["POW.", "PM.", "Marker", "mean", 'EEG.Battery', "CQ.", "EEG.Counter", "EEG.Interpolated", "EEG.RawCq", "timestamp"]

# Seleccionar las columnas que no contengan esos patrones en su nombre
cols_for_corr = [
    col for col in df_eeg.columns
    if not any(keyword in col for keyword in exclusion_keywords)
]

# Filtrar solo esas columnas y quedarnos con las de tipo numérico
df_eeg_filtered = df_eeg[cols_for_corr].select_dtypes(include=[np.number])

# Calcular la matriz de correlación
corr_matrix = df_eeg_filtered.corr()

# Mostrar la matriz de correlación, por ejemplo, como heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
plt.title("Matriz de correlación (excluyendo columnas POW..., PM..., Marker...)")
plt.show()
# Extraer y ordenar las correlaciones
# Usamos la parte superior de la matriz (sin duplicar i-j y j-i)
upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
corr_pairs = corr_matrix.where(upper_triangle).stack()  # Series con (col1, col2) como índice

# Ordenamos por valor absoluto de correlación en orden descendente
# corr_pairs es una Series; reindexamos para obtener el orden según su valor absoluto
corr_pairs_abs_sorted = corr_pairs.reindex(corr_pairs.abs().sort_values(ascending=False).index)

# Imprimir, por ejemplo, las 20 correlaciones más altas
print("=== Top 10 correlaciones (por valor absoluto) excluyendo POW..., PM..., Marker... ===")
for (col1, col2), val in corr_pairs_abs_sorted.head(20).items():
    print(f"{col1} - {col2}: {val:.4f}")
# Diccionario con los DataFrames de ECG
ecg_datasets = {
    "ECG (HR)": df_ecg_hr,
    "ECG (IBI)": df_ecg_ibi
}

for name, df in ecg_datasets.items():
    print(f"=== Análisis para {name} ===")
    
    # Seleccionar solo columnas numéricas
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Calcular la matriz de correlación
    corr_matrix = df_numeric.corr()
    
    # Visualizar la matriz de correlación
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Matriz de correlación - {name}")
    plt.show()
    
    # Extraer las correlaciones (usando solo la parte superior de la matriz para no duplicar)
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    corr_pairs = corr_matrix.where(mask).stack()  # Serie con índices (col1, col2)
    
    # Ordenar las correlaciones por valor absoluto, de mayor a menor
    corr_pairs_sorted = corr_pairs.reindex(corr_pairs.abs().sort_values(ascending=False).index)
    
    # Imprimir las 10 correlaciones más altas
    print(f"Top 10 correlaciones para {name}:")
    for (col1, col2), corr_value in corr_pairs_sorted.head(10).items():
        print(f"{col1} - {col2}: {corr_value:.4f}")
    
    print("-" * 60)
# Excluir las columnas "hr" y "rr_int" respectivamente
df_ecg_hr_comp = df_ecg_hr.drop(columns=['hr'])
df_ecg_ibi_comp = df_ecg_ibi.drop(columns=['rr_int'])

# Verificar si son idénticos (misma posición y mismos datos)
if df_ecg_hr_comp.equals(df_ecg_ibi_comp):
    print("Los DataFrames df_ecg_hr y df_ecg_ibi son idénticos (excluyendo 'hr' y 'rr_int').")
else:
    print("Los DataFrames df_ecg_hr y df_ecg_ibi NO son idénticos (excluyendo 'hr' y 'rr_int').")
    
    # Mostrar diferencias:
    diferencias = df_ecg_hr_comp.compare(df_ecg_ibi_comp)
    print("Diferencias encontradas:")
    print(diferencias)
def check_zeros_in_dataset(df, value_col, group_cols, dataset_name):
    grouped = df.groupby(group_cols)
    for group_key, group in grouped:
        # Ordenamos por datetime para ver la secuencia temporal
        group_sorted = group.sort_values('datetime')
        zero_values = group_sorted[group_sorted[value_col] == 0.000000]
        if not zero_values.empty:
            print(f"Dataset: {dataset_name} - Group: {group_key}")
            print(zero_values[['datetime', value_col]])
            print("-" * 40)

# Verificar en df_ecg_hr para la columna 'hr'
print("Chequeo de valores 0.0 en 'hr' en df_ecg_hr:")
check_zeros_in_dataset(df_ecg_hr, 'hr', ['subject', 'flight', 'phase'], 'ECG (HR)')

# Verificar en df_ecg_ibi para la columna 'rr_int'
print("\nChequeo de valores 0.0 en 'rr_int' en df_ecg_ibi:")
check_zeros_in_dataset(df_ecg_ibi, 'rr_int', ['subject', 'flight', 'phase'], 'ECG (IBI)')
def check_zeros_in_eeg(df, electrode_cols, group_cols, dataset_name):
    # Agrupar según las columnas indicadas
    grouped = df.groupby(group_cols)
    for group_key, group in grouped:
        # Ordenar el grupo por datetime para seguir la secuencia temporal
        group_sorted = group.sort_values('datetime')
        # Filtrar las filas donde alguna de las columnas de electrodos tenga 0.0
        zeros = group_sorted[group_sorted[electrode_cols].eq(0.0).any(axis=1)]
        if not zeros.empty:
            print(f"Dataset: {dataset_name} - Grupo: {group_key}")
            print(zeros[['datetime'] + electrode_cols])
            print("-" * 40)

# lista de columnas de electrodos a comprobar.
electrode_cols = [
    "EEG.AF3", "EEG.F7", "EEG.F3", "EEG.FC5",
    "EEG.F8", "EEG.AF4", "EEG.F4", "EEG.FC6",
    "EEG.T7", "EEG.P7", "EEG.O1", "EEG.T8",
    "EEG.P8", "EEG.O2"
]

print("Chequeo de valores 0.0 en los electrodos en df_eeg:")
check_zeros_in_eeg(df_eeg, electrode_cols, ['subject', 'flight', 'phase'], "EEG")
# Lista de columnas de electrodos
electrode_cols = [
    "EEG.AF3", "EEG.F7", "EEG.F3", "EEG.FC5",
    "EEG.F8", "EEG.AF4", "EEG.F4", "EEG.FC6",
    "EEG.T7", "EEG.P7", "EEG.O1", "EEG.T8",
    "EEG.P8", "EEG.O2"
]

# Filtrar el DataFrame para obtener las filas donde al menos uno de los electrodos es 0.0
df_eeg_zeros = df_eeg[df_eeg[electrode_cols].eq(0.0).any(axis=1)]

# Ordenar por subject, flight, phase y datetime para ver la secuencia temporal
df_eeg_zeros_sorted = df_eeg_zeros.sort_values(by=['subject', 'flight', 'phase', 'datetime'])

# Visualizar el resultado: se muestran la columna datetime y las columnas de electrodos
print("Filas del df_eeg con al menos un valor 0.0 en los electrodos:")
print(df_eeg_zeros_sorted[['datetime'] + electrode_cols])
df_eeg_zeros_sorted
df_eeg_zeros_sorted['datetime']
# Ordenar el dataset completo
df_eeg_sorted = df_eeg.sort_values(by=['subject', 'flight', 'phase', 'datetime']).reset_index(drop=True)

# Ya tenemos df_eeg_zeros_sorted (filas con 0.0) también ordenado
df_zeros = df_eeg_zeros_sorted.reset_index(drop=True)

# Para cada par consecutivo de filas con 0.0
for i in range(len(df_zeros) - 1):
    # Fila i y fila i+1
    row_current = df_zeros.loc[i]
    row_next = df_zeros.loc[i+1]
    
    # Verificamos que sean el mismo sujeto, vuelo y fase
    same_subject = (row_current['subject'] == row_next['subject'])
    same_flight = (row_current['flight'] == row_next['flight'])
    same_phase = (row_current['phase'] == row_next['phase'])
    
    if same_subject and same_flight and same_phase:
        # Intervalo de tiempo: entre la datetime de row_current y row_next
        dt_min = row_current['datetime']
        dt_max = row_next['datetime']
        
        # Extraer las filas del dataset completo que estén dentro de este rango
        mask = (
            (df_eeg_sorted['subject'] == row_current['subject']) &
            (df_eeg_sorted['flight'] == row_current['flight']) &
            (df_eeg_sorted['phase'] == row_current['phase']) &
            (df_eeg_sorted['datetime'] > dt_min) &
            (df_eeg_sorted['datetime'] < dt_max)
        )
        
        df_inbetween = df_eeg_sorted[mask]
        
        if not df_inbetween.empty:
            print(f"Registros intermedios entre las filas {i} y {i+1} con 0.0 (subject={row_current['subject']}, "
                  f"flight={row_current['flight']}, phase={row_current['phase']}):")
            print(df_inbetween[['datetime']])  # Añadir otras columnas que queramos ver
            print("-" * 60)
        else:
            print(f"No hay registros intermedios entre las filas {i} y {i+1} con 0.0 para subject={row_current['subject']}, "
                  f"flight={row_current['flight']}, phase={row_current['phase']}.")
    else:
        print(f"Las filas {i} y {i+1} con 0.0 no comparten subject/flight/phase.")
# Seleccionar las columnas cuyo nombre comienza con "CQ"
cq_cols = [col for col in df_eeg_zeros_sorted.columns if col.startswith("CQ")]
print("Columnas que comienzan con 'CQ':", cq_cols)

# Filtrar las filas donde al menos una de estas columnas tiene valor 0
df_cq0 = df_eeg_zeros_sorted[df_eeg_zeros_sorted[cq_cols].eq(0.000000).any(axis=1)]

# Visualizar los primeros registros resultantes
print("Registros con algún valor 0 en columnas que comienzan con 'CQ':")
print(df_cq0.head())
# Filtrar las filas donde al menos una de estas columnas tiene un valor NaN
df_cq_nan = df_eeg_zeros_sorted[df_eeg_zeros_sorted[cq_cols].isnull().any(axis=1)]

# Mostrar las primeras filas del DataFrame resultante
print("Registros con algún NaN en columnas que comienzan con 'CQ':")
print(df_cq_nan.head())
for name, df in dataframes.items():
    print(f"=== Verificando duplicados en {name} ===")
    
    # Identificar filas duplicadas (keep=False marca todas las duplicadas, no solo la segunda aparición)
    duplicated_mask = df.duplicated(keep=False)
    
    # Contar cuántas filas duplicadas hay
    num_duplicated = duplicated_mask.sum()
    
    if num_duplicated == 0:
        print("No se encontraron filas duplicadas.")
    else:
        print(f"Se encontraron {num_duplicated} filas duplicadas. Mostrando filas duplicadas:")
        duplicates = df[duplicated_mask]
        print(duplicates)
    
    print("-" * 60)
# Verificar duplicados en df_eeg para las claves
print("Duplicados en eeg:")
print(df_eeg.duplicated(subset=['subject', 'flight', 'phase', 'datetime']).sum())
# Verificar duplicados en df_ecg_hr para las claves
print("Duplicados en ecg_hr:")
print(df_ecg_hr.duplicated(subset=['subject', 'flight', 'phase', 'datetime']).sum())

# Verificar duplicados en df_ecg_ibi para las claves
print("Duplicados en ecg_ibi:")
print(df_ecg_ibi.duplicated(subset=['subject', 'flight', 'phase', 'datetime']).sum())
dup_mask = df_ecg_hr.duplicated(subset=['subject','flight','phase','datetime'], keep=False)
df_ecg_hr[dup_mask].sort_values(['subject','flight','phase','datetime']).head(20)
dup_mask = df_ecg_ibi.duplicated(subset=['subject','flight','phase','datetime'], keep=False)
df_ecg_ibi[dup_mask].sort_values(['subject','flight','phase','datetime']).head(20)

# Exportamos a CSV los resultados principales de este script

base_dir = "/Users/danillopis/Desktop/TFG/UAB/workload_dataset"
output_dir = os.path.join(base_dir, "outputs_eda")
os.makedirs(output_dir, exist_ok=True)

# 1) DataFrames con nuevas columnas tras la EDA
df_eeg.to_csv(os.path.join(output_dir, "df_eeg_eda.csv"), index=False)
df_ecg_hr.to_csv(os.path.join(output_dir, "df_ecg_hr_eda.csv"), index=False)
df_ecg_ibi.to_csv(os.path.join(output_dir, "df_ecg_ibi_eda.csv"), index=False)

# Top correlaciones EEG (pares y valores)
corr_pairs_abs_sorted.reset_index().rename(columns={
    "level_0": "var1", "level_1": "var2", 0: "corr_abs"
}).to_csv(os.path.join(output_dir, "top_corr_eeg.csv"), index=False)

# Correlación HR vs dificultad por fase
grouped_corr.reset_index().to_csv(
    os.path.join(output_dir, "corr_by_phase_hr_diff.csv"),
    index=False
)

# Combinaciones faltantes en cada dataset
missing_eeg.to_csv(os.path.join(output_dir, "missing_combinations_eeg.csv"), index=False)
missing_ecg_hr.to_csv(os.path.join(output_dir, "missing_combinations_ecg_hr.csv"), index=False)
missing_ecg_ibi.to_csv(os.path.join(output_dir, "missing_combinations_ecg_ibi.csv"), index=False)

# Filas con al menos un 0.0 en EEG
df_eeg_zeros_sorted.to_csv(
    os.path.join(output_dir, "df_eeg_zeros.csv"),
    index=False
)

# Duplicados detectados en cada dataset
df_eeg[df_eeg.duplicated(keep=False)].to_csv(
    os.path.join(output_dir, "duplicates_eeg.csv"),
    index=False
)
df_ecg_hr[df_ecg_hr.duplicated(keep=False)].to_csv(
    os.path.join(output_dir, "duplicates_ecg_hr.csv"),
    index=False
)
df_ecg_ibi[df_ecg_ibi.duplicated(keep=False)].to_csv(
    os.path.join(output_dir, "duplicates_ecg_ibi.csv"),
    index=False
)

print("EDO completado. CSVs con resultados en:", output_dir)