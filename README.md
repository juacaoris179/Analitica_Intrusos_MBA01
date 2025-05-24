# 🔐 Intrusion Detector App

Este proyecto es una aplicación web creada con **Streamlit** que detecta si una conexión bancaria fue realizada por un intruso o el titular legítimo, usando modelos de Machine Learning entrenados con datos históricos.

---

## 📂 Estructura del repositorio

| Archivo                         | Descripción                                                  |
|--------------------------------|--------------------------------------------------------------|
| `app_intruso.py`               | App Streamlit para cargar datos y predecir intrusos          |
| `modelo_xgb.pkl`               | Modelo XGBoost entrenado y guardado                          |
| `entrenamiento_y_guardado.py` | Script Python que entrena el modelo y guarda el `.pkl`       |
| `requirements.txt`            | Dependencias necesarias para ejecutar la app                 |
| `README.md`                    | Documentación del proyecto                                   |

---

## ⚙️ Cómo usar

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/intrusion-detector-app.git
cd intrusion-detector-app
