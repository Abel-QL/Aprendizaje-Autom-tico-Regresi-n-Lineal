# 📈 Predicción de Precios de Activos Financieros con Regresión Lineal

Proyecto de Machine Learning supervisado para estimar el precio futuro de un ticker bursátil, utilizando distintos modelos de regresión y comparando su rendimiento mediante métricas estándar.

## 🎯 Objetivo

Desarrollar un sistema capaz de predecir el precio futuro de un activo financiero a partir de datos históricos del mercado, entrenando y comparando múltiples modelos de regresión para seleccionar el de mejor capacidad predictiva.

## 🛠️ Tecnologías

- **Python**
- `scikit-learn` — entrenamiento y evaluación de modelos
- `pandas`, `numpy` — manejo y análisis de datos
- `matplotlib` — visualización
- `alpaca-py` — obtención de datos históricos vía Alpaca Markets API

## 🔄 Pipeline del proyecto

1. **Recolección de datos** — Extracción de datos históricos (precios, volumen, indicadores) mediante la API de Alpaca.
2. **Preprocesamiento** — Ingeniería de características, limpieza de datos nulos, codificación de variables categóricas y normalización/estandarización.
3. **Análisis Exploratorio de Datos (EDA)** — Visualizaciones y estadísticas descriptivas para identificar patrones y relaciones entre variables.
4. **Entrenamiento del modelo** — División en conjuntos de entrenamiento/prueba (X, y) y entrenamiento de distintos algoritmos de regresión.
5. **Validación y evaluación** — Comparación de modelos mediante MAE, MSE, RMSE y R².
6. **Predicción con nuevos datos** — Aplicación del modelo entrenado sobre nuevos datos (`.csv`) para generar predicciones.

## 🤖 Modelos implementados

- Ordinary Least Squares Regression
- Ridge Regression
- Bayesian Regression
- Lasso Regression
- Nearest Neighbors Regression
- Random Forest Regression
- Support Vector Machine (SVM) Regression
- Neural Network MLP Regression

## 📊 Dataset

- Mínimo **6 características** relacionadas con el comportamiento del activo financiero.
- Variable objetivo **numérica continua** (precio futuro del ticker).
- Mínimo **800 registros**.

## ✅ Funcionalidades de la entrega

- Notebook (Google Colab / GitHub Codespace) con el código fuente completo.
- Entrenamiento y comparación de los 8 modelos de regresión.
- Visualización de métricas de rendimiento por modelo.
- Módulo para realizar predicciones ingresando un archivo `.csv` con nuevos datos.

## 📄 Documento de infraestructura

El proyecto incluye un documento con:

1. Gráficas analíticas del EDA.
2. Proporción de los conjuntos de entrenamiento y prueba.
3. Descripción de los algoritmos de regresión utilizados.
4. Hiperparámetros empleados en cada modelo.
5. Métricas y estadísticas de rendimiento.
6. Explicación del funcionamiento del sistema de predicción.

## 👨‍🏫 Créditos

Proyecto académico — Aprendizaje Automático
Prof. Ramón E. Álvarez S.
