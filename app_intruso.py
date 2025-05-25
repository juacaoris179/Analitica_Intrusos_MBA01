import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import pickle

# Cargar modelo entrenado (supón que guardaste el modelo XGBoost con pickle)
# modelo = pickle.load(open('modelo_xgb.pkl', 'rb'))
# Para demo, creamos uno falso (simulación)
with open("modelo_xgb.pkl", "rb") as f:
    modelo = pickle.load(f

st.set_page_config(page_title="Detección de Intrusos", layout="centered")
st.title("🔐 Clasificación de Conexiones Fraudulentas")
st.write("Sube un archivo CSV con conexiones o completa los campos manualmente para evaluar la probabilidad de que una conexión sea fraudulenta (realizada por un intruso).")

# === Entrada manual ===
st.sidebar.header("Entrada manual")
ip_extranjera = st.sidebar.selectbox("¿La IP es extranjera?", [0, 1])
minutos = st.sidebar.slider("Minutos de conexión", 0.0, 60.0, 5.0)
conexiones = st.sidebar.slider("N° de conexiones en últimos 3 meses", 0, 100, 3)

if st.sidebar.button("Evaluar manualmente"):
    datos = pd.DataFrame([[ip_extranjera, minutos, conexiones]],
                         columns=['FLAG_IP_EXTRANJERA', 'MINUTOS_CONEXION', 'N_CONEXION_U3M'])
    proba = modelo.predict_proba(datos)[0][1]
    st.metric(label="Probabilidad de intrusión", value=f"{proba:.2%}", delta=None)
    st.progress(proba)

# === Carga de archivo ===
st.subheader("📂 Subir archivo CSV")
archivo = st.file_uploader("Sube tu archivo de datos", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)
    st.write("Vista previa del archivo:", df.head())

    # Verificar columnas esperadas
    columnas_esperadas = {'FLAG_IP_EXTRANJERA', 'MINUTOS_CONEXION', 'N_CONEXION_U3M'}
    if columnas_esperadas.issubset(df.columns):
        proba = modelo.predict_proba(df)[:, 1]
        df['Probabilidad_Intruso'] = proba
        st.success("Predicciones generadas exitosamente")
        st.dataframe(df[['FLAG_IP_EXTRANJERA', 'MINUTOS_CONEXION', 'N_CONEXION_U3M', 'Probabilidad_Intruso']])
        st.bar_chart(df['Probabilidad_Intruso'])
    else:
        st.error("El archivo no contiene las columnas necesarias. Se requieren: FLAG_IP_EXTRANJERA, MINUTOS_CONEXION, N_CONEXION_U3M")

st.caption("Desarrollado por JC | Modelo XGBoost entrenado con SMOTE")
