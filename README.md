# 🔐 Intrusion Detector App

Este proyecto es una aplicación web creada con **Streamlit** que detecta si una conexión bancaria fue realizada por un intruso o por el titular legítimo, utilizando un modelo de Machine Learning entrenado con datos históricos.

---

## 📂 Estructura del repositorio

| Archivo                         | Descripción                                                  |
|--------------------------------|--------------------------------------------------------------|
| `app_intruso.py`               | App Streamlit para evaluar conexiones manuales o por archivo |
| `modelo_rf.pkl`                | Modelo Random Forest entrenado y guardado (`joblib`)         |
| `entrenamiento_y_guardado.py` | Script Python que entrena el modelo y guarda el `.pkl`       |
| `requirements.txt`             | Dependencias necesarias para ejecutar la app                 |
| `README.md`                    | Documentación del proyecto                                   |

---

## ⚙️ Cómo usar

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/intrusion
