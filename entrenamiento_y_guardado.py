
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

# Cargar datos
df = pd.read_csv("Tabla_Intruso_Detectado.csv", sep=';')
df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")

# Validación de columnas
columnas_requeridas = ['FLAG_INTRUSO_DETECTADO', 'FLAG_IP_EXTRANJERA', 'MINUTOS_CONEXION', 'N_CONEXION_U3M']
if not set(columnas_requeridas).issubset(df.columns):
    raise ValueError(f"Columnas faltantes: {set(columnas_requeridas) - set(df.columns)}")

# Crear variable derivada
df['MINUTOS_POR_CONEXION'] = df['MINUTOS_CONEXION'] / (df['N_CONEXION_U3M'] + 1)

# Preparar datos
X = df[['FLAG_IP_EXTRANJERA', 'MINUTOS_CONEXION', 'N_CONEXION_U3M', 'MINUTOS_POR_CONEXION']]
y = df['FLAG_INTRUSO_DETECTADO']

# Aplicar SMOTE
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X, y)

# Dividir entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, stratify=y_sm, random_state=42)

# Entrenar modelo
modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
modelo_rf.fit(X_train, y_train)

# Evaluar
y_pred = modelo_rf.predict(X_test)
y_proba = modelo_rf.predict_proba(X_test)[:, 1]
print("=== Clasificación ===")
print(classification_report(y_test, y_pred))
print("=== Matriz de Confusión ===")
print(confusion_matrix(y_test, y_pred))
print("=== AUC ===")
print(roc_auc_score(y_test, y_proba))

# Guardar modelo
joblib.dump(modelo_rf, "modelo_rf.pkl")
print("✅ Modelo guardado como modelo_rf.pkl")
