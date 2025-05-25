import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Detección de Intrusos", layout="centered")
st.title("🔐 Clasificación de Conexiones Fraudulentas")
st.write("Sube un archivo CSV con conexiones o completa los campos manualmente para evaluar la probabilidad de que una conexión sea fraudulenta (realizada por un intruso).")

# === Cargar y entrenar modelo desde CSV ===
@st.cache_resource
def entrenar_modelo():
    df = pd.read_csv("Tabla_Intruso_Detectado.csv", sep=';')  # <- Aseguramos delimitador correcto
    st.write("Columnas detectadas:", df.columns.tolist())  # Mostrar columnas para debug
    columnas_renombradas = {
        'FLAG_INTRUSO_DETECTADO': 'FLAG_INTRUSO_DETECTADO',
        'FLAG_IP_EXTRANJERA': 'FLAG_IP_EXTRANJERA',
        'MINUTOS_CONEXION': 'MINUTOS_CONEXION',
        'N_CONEXION_U3M': 'N_CONEXION_U3M'
    }
    df = df.rename(columns=columnas_renombradas)
    X = df[['FLAG_IP_EXTRANJERA', 'MINUTOS_CONEXION', 'N_CONEXION_U3M']]
    y = df['FLAG_INTRUSO_DETECTADO']
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X, y)
    X_train, _, y_train, _ = train_test_split(X_sm, y_sm, test_size=0.2, stratify=y_sm, random_state=42)
    modelo = XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=4, learning_rate=0.1, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo

modelo = entrenar_modelo()

# === Entrada manual ===
st.sidebar.header("Entrada manual")
ip_extranjera = st.sidebar.selectbox("¿La IP es extranjera?", [0, 1])
minutos = st.sidebar.slider("Minutos de conexión", 0.0, 60.0, 5.0)
conexiones = st.sidebar.slider("N° de conexiones en últimos 3 meses", 0, 100, 3)

if st.sidebar.button("Evaluar manualmente"):
    datos = pd.DataFrame([[ip_extranjera, minutos, conexiones]],
                         columns=['FLAG_IP_EXTRANJERA', 'MINUTOS_CONEXION', 'N_CONEXION_U3M'])
    proba = float(modelo.predict_proba(datos)[0][1])  # Convertimos a float
    st.metric(label="Probabilidad de intrusión", value=f"{proba:.2%}")
    st.progress(min(proba, 1.0))

# === Carga de archivo ===
st.subheader("📂 Subir archivo CSV")
archivo = st.file_uploader("Sube tu archivo de datos", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo, sep=';')
    df = df.rename(columns=lambda x: x.strip().upper())
    columnas_esperadas = {'FLAG_IP_EXTRANJERA', 'MINUTOS_CONEXION', 'N_CONEXION_U3M'}
    st.write("Columnas cargadas:", df.columns.tolist())

    if columnas_esperadas.issubset(df.columns):
        proba = modelo.predict_proba(df[list(columnas_esperadas)])[:, 1]
        df['Probabilidad_Intruso'] = proba
        st.success("Predicciones generadas exitosamente")
        st.dataframe(df[['FLAG_IP_EXTRANJERA', 'MINUTOS_CONEXION', 'N_CONEXION_U3M', 'Probabilidad_Intruso']])
        st.bar_chart(df['Probabilidad_Intruso'])
    else:
        st.error("El archivo no contiene las columnas necesarias. Se requieren: FLAG_IP_EXTRANJERA, MINUTOS_CONEXION, N_CONEXION_U3M")

st.caption("Desarrollado por JC | Modelo XGBoost entrenado dinámicamente desde CSV")
