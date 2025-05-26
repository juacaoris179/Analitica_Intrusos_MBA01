import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Detección de Intrusos", layout="centered")
st.title("🔐 Clasificación de Conexiones Fraudulentas")
st.write("Evalúa si una conexión bancaria fue realizada por un intruso en base a sus características.")

def score_riesgo_conexion(flag_ip_extranjera, minutos_conexion, n_conexion_u3m):
    score = 0
    if flag_ip_extranjera == 1:
        score += 50
    if minutos_conexion <= 12 and n_conexion_u3m <= 10:
        score += 30
    if n_conexion_u3m <= 3:
        score += 20
    if n_conexion_u3m >= 50:
        score -= 20
    if score >= 70:
        return score, "CRÍTICO"
    elif score >= 40:
        return score, "MODERADO"
    else:
        return score, "BAJO"

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_rf.pkl")

modelo = cargar_modelo()

st.sidebar.header("Entrada manual")
ip_extranjera = st.sidebar.selectbox("¿La IP es extranjera?", [0, 1])
minutos = st.sidebar.slider("Minutos de conexión", 0.0, 60.0, 5.0)
conexiones = st.sidebar.slider("N° de conexiones en últimos 3 meses", 0, 100, 3)

if st.sidebar.button("Evaluar manualmente"):
    datos = pd.DataFrame([[ip_extranjera, minutos, conexiones]],
                         columns=['FLAG_IP_EXTRANJERA', 'MINUTOS_CONEXION', 'N_CONEXION_U3M'])
    proba = float(modelo.predict_proba(datos)[0][1])
    pred = int(modelo.predict(datos)[0])
    score, nivel = score_riesgo_conexion(ip_extranjera, minutos, conexiones)

    st.subheader("🧠 Predicción del Modelo")
    st.write(f"¿Intruso?: {'Sí' if pred == 1 else 'No'} (probabilidad: {proba:.2%})")
    st.progress(min(proba, 1.0))

    st.subheader("📊 Score de Riesgo Operativo")
    st.write(f"Score: **{score}** → Nivel: **{nivel}**")

st.subheader("📂 Subir archivo CSV")
archivo = st.file_uploader("Sube tu archivo de datos", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo, sep=';')
    df.columns = df.columns.str.strip().str.upper()
    columnas_esperadas = ['FLAG_IP_EXTRANJERA', 'MINUTOS_CONEXION', 'N_CONEXION_U3M']

    if set(columnas_esperadas).issubset(df.columns):
        df_input = df[columnas_esperadas]
        df['Probabilidad_Intruso'] = modelo.predict_proba(df_input)[:, 1]
        df['Intruso_Predicho'] = modelo.predict(df_input)

        scores, niveles = [], []
        for _, fila in df_input.iterrows():
            s, n = score_riesgo_conexion(fila['FLAG_IP_EXTRANJERA'],
                                         fila['MINUTOS_CONEXION'],
                                         fila['N_CONEXION_U3M'])
            scores.append(s)
            niveles.append(n)
        df['Score_Riesgo'] = scores
        df['Nivel_Riesgo'] = niveles

        st.success("✅ Predicciones generadas exitosamente")
        st.dataframe(df[columnas_esperadas + ['Probabilidad_Intruso', 'Intruso_Predicho', 'Nivel_Riesgo']])
        st.bar_chart(df['Probabilidad_Intruso'])

    else:
        st.error(f"❌ El archivo debe contener las columnas: {columnas_esperadas}")

st.caption("Desarrollado por JC | Modelo Random Forest entrenado previamente")
