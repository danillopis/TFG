import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch

from collections import Counter
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, classification_report, recall_score, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import matplotlib.pyplot as plt

# Rutas
base_dir = "/Users/danillopis/Desktop/TFG/UAB/workload_dataset"
input_dir = os.path.join(base_dir, "outputs_preproc")
output_dir = os.path.join(base_dir, "outputs_features")
os.makedirs(output_dir, exist_ok=True)
# Cargar DataFrame de características
features_csv_path = os.path.join(output_dir, "features.csv")
# Cargar CSV de características
df_features = pd.read_csv(features_csv_path)

# Detectar columnas constantes
constant_cols = [col for col in df_features.columns if df_features[col].nunique(dropna=False) <= 1]
df_features_clean = df_features.drop(columns=constant_cols)

# Definir X, y
cols_to_drop = [
    "perceived_difficulty", "start_time", "phase", "flight",
    "theoretical_difficulty", "composite_index", "new_cognitive_load_global",
    "subject_median", "new_cognitive_load_subject", "rolling_median",
    "new_cognitive_load_rolling", "subject"
]
X = df_features_clean.drop(columns=cols_to_drop, errors="ignore")
y = df_features_clean["theoretical_difficulty"].copy()

# Métricas personalizadas
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    specs = []
    for i in range(n_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        specs.append(TN / (TN + FP) if (TN + FP) > 0 else 0.0)
    return np.mean(specs)

sens_scorer = make_scorer(recall_score, average="macro")
spec_scorer = make_scorer(specificity_score)

# División entrenamiento/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# -----------------------------
# 2.1) Logistic Regression
# -----------------------------
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=10000, random_state=42))
])
param_grid_lr = {
    "clf__C": [1, 10, 100],
    "clf__solver": ["liblinear", "lbfgs"],
    "clf__class_weight": [None, "balanced"],
}
gs_lr = GridSearchCV(
    pipe_lr, param_grid_lr,
    cv=cv,
    scoring={"sens": sens_scorer, "spec": spec_scorer},
    refit="sens",
    n_jobs=-1,
    error_score="raise"
)
print("Entrenando LogisticRegression con GridSearchCV…")
gs_lr.fit(X_train, y_train)
print("Mejores parámetros LR:", gs_lr.best_params_)

y_pred_lr = gs_lr.predict(X_test)
print("\n— LogisticRegression en TEST —")
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))
print(f"Sensibilidad macro: {recall_score(y_test, y_pred_lr, average='macro'):.3f}")
print(f"Especificidad macro: {specificity_score(y_test, y_pred_lr):.3f}")

# LR + SMOTE + SelectKBest
pipe_lr_smote = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("fs", SelectKBest(f_classif, k=70)),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=30000, random_state=42))
])
gs_lr_smote = GridSearchCV(
    pipe_lr_smote, param_grid_lr,
    cv=cv,
    scoring={"sens": sens_scorer, "spec": spec_scorer},
    refit="sens",
    n_jobs=-1,
    error_score="raise"
)
print("\nEntrenando LogisticRegression + SMOTE + FS…")
gs_lr_smote.fit(X_train, y_train)
print("Mejores parámetros LR (SMOTE+FS):", gs_lr_smote.best_params_)

y_pred_lr_smote = gs_lr_smote.predict(X_test)
print("\n— LogisticRegression (SMOTE + FS) —")
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_lr_smote))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr_smote))
print(f"Sensibilidad macro: {recall_score(y_test, y_pred_lr_smote, average='macro'):.3f}")
print(f"Especificidad macro: {specificity_score(y_test, y_pred_lr_smote):.3f}")

# -----------------------------
# 2.2) K-Nearest Neighbors (K-NN)
# -----------------------------
pipe_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier())
])
param_grid_knn = {
    "clf__n_neighbors": list(range(3, 101, 2)),
    "clf__weights": ["uniform", "distance"],
    "clf__p": [1, 2],
}
gs_knn = GridSearchCV(
    pipe_knn, param_grid_knn,
    cv=cv,
    scoring={"sens": sens_scorer, "spec": spec_scorer},
    refit="sens",
    n_jobs=-1,
    error_score="raise"
)
print("Entrenando K-NN con GridSearchCV…")
gs_knn.fit(X_train, y_train)
print("Mejores parámetros K-NN:", gs_knn.best_params_)

y_pred_knn = gs_knn.predict(X_test)
print("\n— K-NN —")
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))
print(f"Sensibilidad macro: {recall_score(y_test, y_pred_knn, average='macro'):.3f}")
print(f"Especificidad macro: {specificity_score(y_test, y_pred_knn):.3f}")

# K-NN + SMOTE + FS
pipe_knn_smote = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("fs", SelectKBest(f_classif, k=70)),
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier())
])
gs_knn_smote = GridSearchCV(
    pipe_knn_smote, param_grid_knn,
    cv=cv,
    scoring={"sens": sens_scorer, "spec": spec_scorer},
    refit="sens",
    n_jobs=-1,
    error_score="raise"
)
print("\nEntrenando K-NN + SMOTE + FS…")
gs_knn_smote.fit(X_train, y_train)
print("Mejores parámetros K-NN (SMOTE+FS):", gs_knn_smote.best_params_)

y_pred_knn_smote = gs_knn_smote.predict(X_test)
print("\n— K-NN (SMOTE + FS) —")
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_knn_smote))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn_smote))
print(f"Sensibilidad macro: {recall_score(y_test, y_pred_knn_smote, average='macro'):.3f}")
print(f"Especificidad macro: {specificity_score(y_test, y_pred_knn_smote):.3f}")

# -----------------------------
# 2.3) Random Forest
# -----------------------------
pipe_rf = Pipeline([
    ("clf", RandomForestClassifier(random_state=42))
])
param_grid_rf = {
    "clf__n_estimators": [50, 100, 200, 500],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__class_weight": [None, "balanced"],
}
gs_rf = GridSearchCV(
    pipe_rf, param_grid_rf,
    cv=cv,
    scoring={"sens": sens_scorer, "spec": spec_scorer},
    refit="sens",
    n_jobs=-1,
    error_score="raise"
)
print("Entrenando RandomForest con GridSearchCV…")
gs_rf.fit(X_train, y_train)
print("Mejores parámetros RF:", gs_rf.best_params_)

y_pred_rf = gs_rf.predict(X_test)
print("\n— RandomForest —")
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print(f"Sensibilidad macro: {recall_score(y_test, y_pred_rf, average='macro'):.3f}")
print(f"Especificidad macro: {specificity_score(y_test, y_pred_rf):.3f}")

# RF + SMOTE
pipe_rf_smote = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf", RandomForestClassifier(random_state=42))
])
gs_rf_smote = GridSearchCV(
    pipe_rf_smote, param_grid_rf,
    cv=cv,
    scoring={"sens": sens_scorer, "spec": spec_scorer},
    refit="sens",
    n_jobs=-1,
    error_score="raise"
)
print("\nEntrenando RandomForest + SMOTE…")
gs_rf_smote.fit(X_train, y_train)
print("Mejores parámetros RF (SMOTE):", gs_rf_smote.best_params_)

y_pred_rf_smote = gs_rf_smote.predict(X_test)
print("\n— RandomForest (SMOTE) —")
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_rf_smote))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf_smote))
print(f"Sensibilidad macro: {recall_score(y_test, y_pred_rf_smote, average='macro'):.3f}")
print(f"Especificidad macro: {specificity_score(y_test, y_pred_rf_smote):.3f}")

# Top 10 características RF + SMOTE
best_rf = gs_rf_smote.best_estimator_.named_steps["clf"]
feature_names_rf = X.columns
importances_rf = best_rf.feature_importances_
feature_importances_rf = (
    pd.DataFrame({"Feature": feature_names_rf, "Importance": importances_rf})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)
print("Top 10 características más importantes (Random Forest + SMOTE):\n")
print(feature_importances_rf.head(10))

plt.figure(figsize=(10, 6))
plt.barh(
    feature_importances_rf["Feature"].head(10)[::-1],
    feature_importances_rf["Importance"].head(10)[::-1]
)
plt.xlabel("Importancia")
plt.title("Top 10 características (Random Forest + SMOTE)")
plt.tight_layout()
plt.show()

# -----------------------------
# 2.4) Support Vector Machine (SVM)
# -----------------------------
pipe_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC())
])
param_grid_svm = {
    "clf__C": [10, 100, 1000],
    "clf__kernel": ["linear", "rbf"],
    "clf__gamma": ["scale", "auto"],
    "clf__class_weight": [None, "balanced"],
}
gs_svm = GridSearchCV(
    pipe_svm, param_grid_svm,
    cv=cv,
    scoring={"sens": sens_scorer, "spec": spec_scorer},
    refit="sens",
    n_jobs=-1,
    error_score="raise"
)
print("Entrenando SVM con GridSearchCV…")
gs_svm.fit(X_train, y_train)
print("Mejores parámetros SVM:", gs_svm.best_params_)

y_pred_svm = gs_svm.predict(X_test)
print("\n— SVM —")
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_svm))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))
print(f"Sensibilidad macro: {recall_score(y_test, y_pred_svm, average='macro'):.3f}")
print(f"Especificidad macro: {specificity_score(y_test, y_pred_svm):.3f}")

# SVM + SMOTE + FS
pipe_svm_smote = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("fs", SelectKBest(f_classif, k=70)),
    ("scaler", StandardScaler()),
    ("clf", SVC())
])
gs_svm_smote = GridSearchCV(
    pipe_svm_smote, param_grid_svm,
    cv=cv,
    scoring={"sens": sens_scorer, "spec": spec_scorer},
    refit="sens",
    n_jobs=-1,
    error_score="raise"
)
print("\nEntrenando SVM + SMOTE + FS…")
gs_svm_smote.fit(X_train, y_train)
print("Mejores parámetros SVM (SMOTE+FS):", gs_svm_smote.best_params_)

y_pred_svm_smote = gs_svm_smote.predict(X_test)
print("\n— SVM (SMOTE + FS) —")
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_svm_smote))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm_smote))
print(f"Sensibilidad macro: {recall_score(y_test, y_pred_svm_smote, average='macro'):.3f}")
print(f"Especificidad macro: {specificity_score(y_test, y_pred_svm_smote):.3f}")

# -----------------------------
# 2.5) XGBoost
# -----------------------------
pipe_xgb1 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", XGBClassifier(eval_metric="mlogloss", random_state=42))
])
param_grid_xgb1 = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [3, 5, 7],
    "clf__learning_rate": [0.01, 0.1, 0.2],
    "clf__subsample": [0.8, 1.0],
}
gs_xgb1 = GridSearchCV(
    pipe_xgb1, param_grid_xgb1,
    cv=cv,
    scoring={"sens": sens_scorer, "spec": spec_scorer},
    refit="sens",
    n_jobs=-1,
    error_score="raise"
)
print("Entrenando XGBoost con GridSearchCV…")
gs_xgb1.fit(X_train, y_train)
print("Mejores parámetros XGBoost:", gs_xgb1.best_params_)

y_pred_xgb1 = gs_xgb1.predict(X_test)
print("\n— XGBoost —")
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_xgb1))
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb1))
print(f"Sensibilidad macro: {recall_score(y_test, y_pred_xgb1, average='macro'):.3f}")
print(f"Especificidad macro: {specificity_score(y_test, y_pred_xgb1):.3f}")

# XGBoost + SMOTE + FS
pipe_xgb2 = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("scaler", StandardScaler()),
    ("clf", XGBClassifier(eval_metric="mlogloss", random_state=42))
])
param_grid_xgb2 = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [3, 5, 7],
    "clf__learning_rate": [0.01, 0.1, 0.2],
    "clf__subsample": [0.8, 1.0],
}
gs_xgb2 = GridSearchCV(
    pipe_xgb2, param_grid_xgb2,
    cv=cv,
    scoring={"sens": sens_scorer, "spec": spec_scorer},
    refit="sens",
    n_jobs=-1,
    error_score="raise"
)
print("Entrenando XGBoost + SMOTE + FS…")
gs_xgb2.fit(X_train, y_train)
print("Mejores parámetros XGBoost (SMOTE+FS):", gs_xgb2.best_params_)

y_pred_xgb2 = gs_xgb2.predict(X_test)
print("\n— XGBoost (SMOTE + FS) —")
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_xgb2))
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb2))
print(f"Sensibilidad macro: {recall_score(y_test, y_pred_xgb2, average='macro'):.3f}")
print(f"Especificidad macro: {specificity_score(y_test, y_pred_xgb2):.3f}")

# Top 10 características XGBoost + SMOTE + FS
best_xgb = gs_xgb2.best_estimator_.named_steps["clf"]
feature_names_xgb = X.columns
importances_xgb = best_xgb.feature_importances_
feature_importances_xgb = (
    pd.DataFrame({"Feature": feature_names_xgb, "Importance": importances_xgb})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)
print("Top 10 características más importantes (XGBoost + SMOTE + FS):\n")
print(feature_importances_xgb.head(10))

plt.figure(figsize=(10, 6))
plt.barh(
    feature_importances_xgb["Feature"].head(10)[::-1],
    feature_importances_xgb["Importance"].head(10)[::-1]
)
plt.xlabel("Importancia")
plt.title("Top 10 características (XGBoost + SMOTE + FS)")
plt.tight_layout()
plt.show()

# -----------------------------
# 2.6) Gradient Boosting
# -----------------------------
pipe_gb1 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GradientBoostingClassifier(random_state=42))
])
param_grid_gb1 = {
    "clf__n_estimators": [200, 300],
    "clf__learning_rate": [0.1, 0.2],
    "clf__max_depth": [3, 5, 7],
    "clf__subsample": [0.8, 1.0],
    "clf__max_features": ["sqrt", "log2"],
}
gs_gb1 = GridSearchCV(
    pipe_gb1, param_grid_gb1,
    cv=cv,
    scoring={"sens": sens_scorer, "spec": spec_scorer},
    refit="sens",
    n_jobs=-1,
    error_score="raise"
)
print("Entrenando GradientBoostingClassifier con GridSearchCV…")
gs_gb1.fit(X_train, y_train)
print("Mejores parámetros GB:", gs_gb1.best_params_)

y_pred_gb1 = gs_gb1.predict(X_test)
print("\n— Gradient Boosting —")
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_gb1))
print("\nClassification Report:\n", classification_report(y_test, y_pred_gb1))
print(f"Sensibilidad macro: {recall_score(y_test, y_pred_gb1, average='macro'):.3f}")
print(f"Especificidad macro: {specificity_score(y_test, y_pred_gb1):.3f}")

# GB + SMOTE
pipe_gb2 = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("scaler", StandardScaler()),
    ("clf", GradientBoostingClassifier(random_state=42))
])
param_grid_gb2 = {
    "clf__n_estimators": [200, 300],
    "clf__learning_rate": [0.1, 0.2],
    "clf__max_depth": [3, 5, 7],
    "clf__subsample": [0.8, 1.0],
    "clf__max_features": ["sqrt", "log2"],
}
gs_gb2 = GridSearchCV(
    pipe_gb2, param_grid_gb2,
    cv=cv,
    scoring={"sens": sens_scorer, "spec": spec_scorer},
    refit="sens",
    n_jobs=-1,
    error_score="raise"
)
print("Entrenando GradientBoostingClassifier + SMOTE…")
gs_gb2.fit(X_train, y_train)
print("Mejores parámetros GB (SMOTE):", gs_gb2.best_params_)

y_pred_gb2 = gs_gb2.predict(X_test)
print("\n— Gradient Boosting (SMOTE) —")
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_gb2))
print("\nClassification Report:\n", classification_report(y_test, y_pred_gb2))
print(f"Sensibilidad macro: {recall_score(y_test, y_pred_gb2, average='macro'):.3f}")
print(f"Especificidad macro: {specificity_score(y_test, y_pred_gb2):.3f}")

# Top 10 características GB + SMOTE
best_gb = gs_gb2.best_estimator_.named_steps["clf"]
importances_gb = best_gb.feature_importances_
feature_names_gb = X.columns
feature_importances_gb = (
    pd.DataFrame({"Feature": feature_names_gb, "Importance": importances_gb})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)
print("Top 10 características (Gradient Boosting + SMOTE):\n")
print(feature_importances_gb.head(10))

plt.figure(figsize=(10, 6))
plt.barh(
    feature_importances_gb["Feature"].head(10)[::-1],
    feature_importances_gb["Importance"].head(10)[::-1]
)
plt.xlabel("Importancia")
plt.title("Top 10 características (Gradient Boosting + SMOTE)")
plt.tight_layout()
plt.show()