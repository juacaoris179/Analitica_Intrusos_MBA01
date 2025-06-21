import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Detección de Intrusos", layout="centered")
st.title("🔐 Clasificación de Conexiones Fraudulentas")
st.markdown("Evalúa si una conexión bancaria fue realizada por un intruso en base a sus características.")
st.markdown("---")

# ===== FUNCIONES DE RIESGO Y PREVENCIÓN =====
def score_riesgo_conexion(flag_ip_extranjera, minutos_conexion, n_conexion_u3m):
    score = 0
    razones = []

    if flag_ip_extranjera == 1:
        score += 40
        razones.append("IP extranjera")

    if minutos_conexion <= 10:
        score += 30
        razones.append("Conexión corta (<=10min)")
    elif minutos_conexion <= 15:
        score += 15
        razones.append("Conexión moderadamente corta")

    if n_conexion_u3m == 0:
        score += 10
        razones.append("Sin actividad previa")
    elif n_conexion_u3m <= 3:
        score += 10
        razones.append("Actividad baja")
    elif n_conexion_u3m >= 50:
        score -= 20
        razones.append("Alta actividad")

    if score >= 70:
        nivel = "CRÍTICO"
        color = "🔴"
    elif score >= 40:
        nivel = "MODERADO"
        color = "🟠"
    else:
        nivel = "BAJO"
        color = "🟢"

    return score, nivel, color, razones

def estrategia_prevencion(prob):
    if prob >= 0.90:
        return "🔴 BLOQUEO AUTOMÁTICO"
    elif prob >= 0.70:
        return "🟠 MFA + TICKET AUTOMÁTICO"
    elif prob >= 0.40:
        return "🟡 ALERTA Y MONITOREO"
    elif prob >= 0.20:
        return "🔵 SOLO REGISTRO"
    else:
        return "🟢 NINGUNA ACCIÓN"

# ===== CARGA DEL MODELO =====
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_rf.pkl")

modelo = cargar_modelo()

# ===== ENTRADA MANUAL =====
st.sidebar.header("🧾 Entrada manual")
ip_extranjera = st.sidebar.selectbox("¿La IP es extranjera?", [0, 1])
minutos = st.sidebar.slider("Minutos de conexión", 0.0, 60.0, 5.0)
conexiones = st.sidebar.slider("N° de conexiones en últimos 3 meses", 0, 100, 3)

if st.sidebar.button("Evaluar manualmente"):
    minutos_por_conexion = minutos / (conexiones + 1)
    datos = pd.DataFrame([[ip_extranjera, minutos, conexiones, minutos_por_conexion]],
                         columns=['FLAG_IP_EXTRANJERA', 'MINUTOS_CONEXION', 'N_CONEXION_U3M', 'MINUTOS_POR_CONEXION'])
    
    proba = float(modelo.predict_proba(datos)[0][1])
    pred = int(modelo.predict(datos)[0])
    score, nivel, color, razones = score_riesgo_conexion(ip_extranjera, minutos, conexiones)
    accion = estrategia_prevencion(proba)

    st.subheader("🧠 Predicción del Modelo")
    st.markdown("### ✅ Resultado del Modelo")
    st.markdown(f"**¿Intruso?**: {'🛑 Sí' if pred == 1 else '✅ No'}")
    st.markdown(f"**Probabilidad de intrusión:** `{proba:.2%}`")
    st.progress(min(proba, 1.0))
    st.markdown("---")

    st.subheader("📊 Score de Riesgo Operativo")
    st.markdown(f"**Score:** `{score}` → **Nivel:** {color} **{nivel}**")
    st.markdown("🧾 **Razones:** " + ", ".join(razones))
    st.markdown("---")

    st.subheader("🛡️ Acción Preventiva Recomendada")
    st.markdown(f"**Estrategia de Prevención:** {accion}")
    st.markdown("---")

# ===== SUBIR ARCHIVO CSV =====
st.subheader("📂 Subir archivo CSV")
archivo = st.file_uploader("Sube tu archivo de datos", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo, sep=';')
    df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
    columnas_esperadas = ['FLAG_IP_EXTRANJERA', 'MINUTOS_CONEXION', 'N_CONEXION_U3M']

    if set(columnas_esperadas).issubset(df.columns):
        df["MINUTOS_POR_CONEXION"] = df["MINUTOS_CONEXION"] / (df["N_CONEXION_U3M"] + 1)
        df_input = df[columnas_esperadas + ['MINUTOS_POR_CONEXION']]

        df['Probabilidad_Intruso'] = modelo.predict_proba(df_input)[:, 1]
        df['Intruso_Predicho'] = modelo.predict(df_input)

        scores, niveles, razones_list, acciones = [], [], [], []
        for _, fila in df.iterrows():
            s, n, c, r = score_riesgo_conexion(
                fila['FLAG_IP_EXTRANJERA'],
                fila['MINUTOS_CONEXION'],
                fila['N_CONEXION_U3M']
            )
            scores.append(s)
            niveles.append(f"{c} {n}")
            razones_list.append(", ".join(r))
            acciones.append(estrategia_prevencion(fila['Probabilidad_Intruso']))

        df['Score_Riesgo'] = scores
        df['Nivel_Riesgo'] = niveles
        df['Razones_Riesgo'] = razones_list
        df['Accion_Preventiva'] = acciones

        st.success("✅ Predicciones generadas exitosamente")
        st.dataframe(df[columnas_esperadas + [
            'Probabilidad_Intruso', 'Intruso_Predicho',
            'Nivel_Riesgo', 'Razones_Riesgo', 'Accion_Preventiva'
        ]])
        st.bar_chart(df['Probabilidad_Intruso'])
    else:
        st.error(f"❌ El archivo debe contener las columnas: {columnas_esperadas}")

st.caption("Desarrollado por JC | Modelo Random Forest refinado con variable MINUTOS_POR_CONEXION")


