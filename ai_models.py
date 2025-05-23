import os
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    confusion_matrix, classification_report,
    recall_score, make_scorer
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Directorios
dir_base = "/Users/danillopis/Desktop/TFG/UAB/workload_dataset"
dir_in = os.path.join(dir_base, "outputs_cognitive_load")
dir_out = os.path.join(dir_base, "outputs_models")
os.makedirs(dir_out, exist_ok=True)

# Carga de features con carga cognitiva
df = pd.read_csv(os.path.join(dir_in, "features_cognitive_load.csv"))

# Detectar y eliminar columnas constantes
constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
df = df.drop(columns=constant_cols)

# Definir X, y eliminando columnas irrelevantes
drop_cols = [
    'perceived_difficulty', 'start_time', 'phase', 'flight',
    'theoretical_difficulty', 'composite_index', 'new_cognitive_load_global',
    'subject_median', 'new_cognitive_load_subject', 'rolling_median',
    'new_cognitive_load_rolling'
]
X = df.drop(columns=drop_cols, errors='ignore')
y = df['theoretical_difficulty']

# Función de especificidad macro
from sklearn.metrics import confusion_matrix

def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specs = []
    for i in range(cm.shape[0]):
        TP = cm[i,i]
        FP = cm[:,i].sum() - TP
        FN = cm[i,:].sum() - TP
        TN = cm.sum() - (TP+FP+FN)
        specs.append(TN/(TN+FP) if (TN+FP)>0 else 0.0)
    return np.mean(specs)

sens_scorer = make_scorer(recall_score, average='macro')
spec_scorer = make_scorer(specificity_score)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 1) Logistic Regression
pipe1 = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=10000, random_state=42))
])
param1 = {
    'clf__C': [0.1,1,10,100],
    'clf__solver': ['liblinear','lbfgs','saga','newton-cg'],
    'clf__class_weight': [None,'balanced']
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs1 = GridSearchCV(pipe1, param1, cv=cv,
                   scoring={'sens':sens_scorer,'spec':spec_scorer},
                   refit='sens', n_jobs=-1, error_score='raise')
print("Training Logistic Regression...")
gs1.fit(X_train, y_train)
print("Best LR params:", gs1.best_params_)

y_pred = gs1.predict(X_test)
print("LogReg Confusion Matrix:\n", confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("LR Sens:", recall_score(y_test,y_pred,average='macro'),
      "LR Spec:", specificity_score(y_test,y_pred))

# 1b) LR + SMOTE + FS
pipe2 = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('fs', SelectKBest(f_classif, k=70)),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=30000, random_state=42))
])
gs2 = GridSearchCV(pipe2, param1, cv=cv,
                   scoring={'sens':sens_scorer,'spec':spec_scorer},
                   refit='sens', n_jobs=-1, error_score='raise')
print("Training LR+SMOTE+FS...")
gs2.fit(X_train, y_train)
print("Best LR+SMOTE params:", gs2.best_params_)
y_pred2 = gs2.predict(X_test)
print("LR+SMOTE Confusion Matrix:\n", confusion_matrix(y_test,y_pred2))
print(classification_report(y_test,y_pred2))
print("LR+SMOTE Sens:", recall_score(y_test,y_pred2,average='macro'),
      "LR+SMOTE Spec:", specificity_score(y_test,y_pred2))

# 2) K-NN
pipe_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier())
])
param_knn = {'clf__n_neighbors': list(range(3,101,2)),
             'clf__weights':['uniform','distance'], 'clf__p':[1,2]}
gs_knn = GridSearchCV(pipe_knn, param_knn, cv=cv,
                      scoring={'sens':sens_scorer,'spec':spec_scorer},
                      refit='sens', n_jobs=-1, error_score='raise')
print("Training K-NN...")
gs_knn.fit(X_train, y_train)
print("Best KNN params:", gs_knn.best_params_)
y_knn = gs_knn.predict(X_test)
print("KNN Confusion Matrix:\n", confusion_matrix(y_test,y_knn))
print(classification_report(y_test,y_knn))
print("KNN Sens:", recall_score(y_test,y_knn,average='macro'),
      "KNN Spec:", specificity_score(y_test,y_knn))

# 2b) K-NN + SMOTE + FS
pipe_knn2 = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('fs', SelectKBest(f_classif, k=70)),
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier())
])
gs_knn2 = GridSearchCV(pipe_knn2, param_knn, cv=cv,
                       scoring={'sens':sens_scorer,'spec':spec_scorer},
                       refit='sens', n_jobs=-1, error_score='raise')
print("Training K-NN+SMOTE+FS...")
gs_knn2.fit(X_train, y_train)
print("Best KNN+SMOTE params:", gs_knn2.best_params_)
y_knn2 = gs_knn2.predict(X_test)
print("KNN+SMOTE Confusion Matrix:\n", confusion_matrix(y_test,y_knn2))
print(classification_report(y_test,y_knn2))
print("KNN+SMOTE Sens:", recall_score(y_test,y_knn2,average='macro'),
      "KNN+SMOTE Spec:", specificity_score(y_test,y_knn2))

# 3) Random Forest
pipe_rf = Pipeline([('clf', RandomForestClassifier(random_state=42))])
param_rf = {
    'clf__n_estimators':[50,100,200,500],'clf__max_depth':[None,10,20],
    'clf__min_samples_split':[2,5,10],'clf__min_samples_leaf':[1,2,4],
    'clf__class_weight':[None,'balanced']
}
gs_rf = GridSearchCV(pipe_rf, param_rf, cv=cv,
                     scoring={'sens':sens_scorer,'spec':spec_scorer},
                     refit='sens', n_jobs=-1, error_score='raise')
print("Training RF...")
gs_rf.fit(X_train, y_train)
print("Best RF params:", gs_rf.best_params_)
y_rf = gs_rf.predict(X_test)
print("RF Confusion Matrix:\n", confusion_matrix(y_test,y_rf))
print(classification_report(y_test,y_rf))
print("RF Sens:", recall_score(y_test,y_rf,average='macro'),
      "RF Spec:", specificity_score(y_test,y_rf))

# 3b) RF + SMOTE
pipe_rf2 = ImbPipeline([('smote', SMOTE(random_state=42)),('clf', RandomForestClassifier(random_state=42))])
gs_rf2 = GridSearchCV(pipe_rf2, param_rf, cv=cv,
                      scoring={'sens':sens_scorer,'spec':spec_scorer},
                      refit='sens', n_jobs=-1, error_score='raise')
print("Training RF+SMOTE...")
gs_rf2.fit(X_train, y_train)
print("Best RF+SMOTE params:", gs_rf2.best_params_)
y_rf2 = gs_rf2.predict(X_test)
print("RF+SMOTE Confusion Matrix:\n", confusion_matrix(y_test,y_rf2))
print(classification_report(y_test,y_rf2))
print("RF+SMOTE Sens:", recall_score(y_test,y_rf2,average='macro'),
      "RF+SMOTE Spec:", specificity_score(y_test,y_rf2))

# 4) SVM
pipe_svm = Pipeline([('scaler', StandardScaler()),('clf', SVC())])
param_svm = {
    'clf__C':[0.1,1,10,100,1000],'clf__kernel':['linear','rbf','poly'],
    'clf__gamma':['scale','auto'],'clf__class_weight':[None,'balanced']
}
gs_svm = GridSearchCV(pipe_svm, param_svm, cv=cv,
                      scoring={'sens':sens_scorer,'spec':spec_scorer},
                      refit='sens', n_jobs=-1, error_score='raise')
print("Training SVM...")
gs_svm.fit(X_train, y_train)
print("Best SVM params:", gs_svm.best_params_)
y_svm = gs_svm.predict(X_test)
print("SVM Confusion Matrix:\n", confusion_matrix(y_test,y_svm))
print(classification_report(y_test,y_svm))
print("SVM Sens:", recall_score(y_test,y_svm,average='macro'),
      "SVM Spec:", specificity_score(y_test,y_svm))

# 4b) SVM+SMOTE+FS
pipe_svm2 = ImbPipeline([('smote', SMOTE(random_state=42)),('fs',SelectKBest(f_classif,k=70)),('scaler',StandardScaler()),('clf',SVC())])
gs_svm2 = GridSearchCV(pipe_svm2, param_svm, cv=cv,
                       scoring={'sens':sens_scorer,'spec':spec_scorer},
                       refit='sens', n_jobs=-1, error_score='raise')
print("Training SVM+SMOTE+FS...")
gs_svm2.fit(X_train, y_train)
print("Best SVM+SMOTE params:", gs_svm2.best_params_)
y_svm2 = gs_svm2.predict(X_test)
print("SVM+SMOTE Confusion Matrix:\n", confusion_matrix(y_test,y_svm2))
print(classification_report(y_test,y_svm2))
print("SVM+SMOTE Sens:", recall_score(y_test,y_svm2,average='macro'),
      "SVM+SMOTE Spec:", specificity_score(y_test,y_svm2))

# 5) XGBoost
pipe_xgb1 = Pipeline([('scaler', StandardScaler()),('clf', XGBClassifier(eval_metric='mlogloss', random_state=42))])
param_xgb1 = {'clf__n_estimators':[100,200],'clf__max_depth':[3,5,7],'clf__learning_rate':[0.01,0.1,0.2],'clf__subsample':[0.8,1.0],'clf__colsample_bytree':[0.8,1.0]}
gs_xgb1 = GridSearchCV(pipe_xgb1, param_xgb1, cv=cv,
                        scoring={'sens':sens_scorer,'spec':spec_scorer},
                        refit='sens', n_jobs=-1, error_score='raise')
print("Training XGBoost...")
gs_xgb1.fit(X_train, y_train)
print("Best XGB params:", gs_xgb1.best_params_)
print("\n— Resultados XGBoost —")
y_pred_test = gs_xgb1.predict(X_test)
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred_test))
print(f"Sensibilidad macro: {recall_score(y_test, y_pred_test, average='macro'):.3f}")
print(f"Especificidad macro: {specificity_score(y_test, y_pred_test):.3f}")
pipe_xgb2 = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('clf',   XGBClassifier(
        eval_metric='mlogloss',
        random_state=42
    ))
])

param_grid_xgb2 = {
    'clf__n_estimators':    [100, 200],
    'clf__max_depth':       [3, 5, 7],
    'clf__learning_rate':   [0.01, 0.1, 0.2],
    'clf__subsample':       [0.8, 1.0],
    'clf__colsample_bytree':[0.8, 1.0],
}

gs_xgb2 = GridSearchCV(
    pipe_xgb2, param_grid_xgb2,
    cv=cv,
    scoring={'sens': sens_scorer, 'spec': spec_scorer},
    refit='sens',
    n_jobs=-1,
    error_score='raise'
)
print("\nEntrenando XGBoost + SMOTE + FS…")
gs_xgb2.fit(X_train, y_train)
print("Mejores parámetros (SMOTE+FS):", gs_xgb2.best_params_)

print("\n— Resultados XGBoost (con SMOTE + FS) —")
y_pred_test = gs_xgb2.predict(X_test)
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred_test))
print(f"Sensibilidad macro: {recall_score(y_test, y_pred_test, average='macro'):.3f}")
print(f"Especificidad macro: {specificity_score(y_test, y_pred_test):.3f}")
import pandas as pd
import matplotlib.pyplot as plt

best_pipeline = gs_xgb2.best_estimator_
best_xgb      = best_pipeline.named_steps['clf']

feature_names = X.columns

importances = best_xgb.feature_importances_

feature_importances = (
    pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    .sort_values('Importance', ascending=False)
    .reset_index(drop=True)
)

print("Top 10 características más importantes (XGBoost):\n")
print(feature_importances.head(10))

plt.figure(figsize=(10, 6))
plt.barh(
    feature_importances['Feature'].head(10)[::-1],
    feature_importances['Importance'].head(10)[::-1]
)
plt.xlabel('Importancia')
plt.title('Top 10 características (XGBoost)')
plt.tight_layout()
plt.show()
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, recall_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe_gb1 = Pipeline([
    ('scaler', StandardScaler()),  # opcional en árboles, pero consistente
    ('clf', GradientBoostingClassifier(random_state=42))
])

param_grid_gb1 = {
    'clf__n_estimators': [200, 300],
    'clf__learning_rate': [0.1, 0.2],
    'clf__max_depth': [3, 5, 7],
    'clf__subsample': [0.8, 1.0],
    'clf__max_features': ['sqrt', 'log2']
}

gs_gb1 = GridSearchCV(
    pipe_gb1, param_grid_gb1,
    cv=cv,
    scoring={'sens': sens_scorer, 'spec': spec_scorer},
    refit='sens',
    n_jobs=-1,
    error_score='raise'
)
print("Entrenando GradientBoostingClassifier con GridSearchCV…")
gs_gb1.fit(X_train, y_train)
print("Mejores parámetros:", gs_gb1.best_params_)

print("\n— Resultados Gradient Boosting —")
y_pred_test = gs_gb1.predict(X_test)
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred_test))
print(f"Sensibilidad macro: {recall_score(y_test, y_pred_test, average='macro'):.3f}")
print(f"Especificidad macro: {specificity_score(y_test, y_pred_test):.3f}")
pipe_gb2 = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('clf', GradientBoostingClassifier(random_state=42))
])

param_grid_gb2 = {
    'clf__n_estimators': [200, 300],
    'clf__learning_rate': [0.1, 0.2],
    'clf__max_depth': [3, 5, 7],
    'clf__subsample': [0.8, 1.0],
    'clf__max_features': ['sqrt', 'log2']
}

gs_gb2 = GridSearchCV(
    pipe_gb2, param_grid_gb2,
    cv=cv,
    scoring={'sens': sens_scorer, 'spec': spec_scorer},
    refit='sens',
    n_jobs=-1,
    error_score='raise'
)
print("\nEntrenando GradientBoostingClassifier + SMOTE…")
gs_gb2.fit(X_train, y_train)
print("Mejores parámetros (SMOTE):", gs_gb2.best_params_)

print("\n— Resultados Gradient Boosting (con SMOTE) —")
y_pred_test = gs_gb2.predict(X_test)
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred_test))
print(f"Sensibilidad macro: {recall_score(y_test, y_pred_test, average='macro'):.3f}")
print(f"Especificidad macro: {specificity_score(y_test, y_pred_test):.3f}")
best_gb_basic = gs_gb2.best_estimator_.named_steps['clf']

importances_basic = best_gb_basic.feature_importances_

feature_names = X.columns

feat_imp_basic = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances_basic
}).sort_values(by='Importance', ascending=False)

print("Top 10 características (Gradient Boosting básico):")
print(feat_imp_basic.head(10))

plt.figure(figsize=(8,6))
plt.barh(feat_imp_basic['Feature'].head(10)[::-1], 
         feat_imp_basic['Importance'].head(10)[::-1])
plt.xlabel('Importancia')
plt.title('Top 10 Características (GB básico)')
plt.tight_layout()
plt.show()
