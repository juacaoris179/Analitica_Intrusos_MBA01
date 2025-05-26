# entrenamiento_y_guardado.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import pickle

# === 1. Cargar datos con delimitador correcto ===
df = pd.read_csv("Tabla_Intruso_Detectado.csv", sep=';')
df.columns = df.columns.str.strip().str.upper()

# === 2. Separar variables ===
X = df[['FLAG_IP_EXTRANJERA', 'MINUTOS_CONEXION', 'N_CONEXION_U3M']]
y = df['FLAG_INTRUSO_DETECTADO']

# === 3. Balanceo con SMOTE ===
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X, y)

# === 4. División entrenamiento / prueba ===
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, stratify=y_sm, random_state=42)

# === 5. Entrenamiento del modelo ===
modelo_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=4, learning_rate=0.1, random_state=42)
modelo_xgb.fit(X_train, y_train)

# === 6. Evaluación ===
y_pred = modelo_xgb.predict(X_test)
y_proba = modelo_xgb.predict_proba(X_test)[:, 1]

print("\n=== Reporte de Clasificación ===")
print(classification_report(y_test, y_pred))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nAUC:", roc_auc_score(y_test, y_proba))

# === 7. Guardar modelo entrenado ===
with open("modelo_xgb.pkl", "wb") as f:
    pickle.dump(modelo_xgb, f)

print("\n✅ Modelo guardado como modelo_xgb.pkl")
