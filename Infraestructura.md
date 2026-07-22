# Documento de Infraestructura del Proyecto

## Predicción de Precios de Meta Platforms (META) mediante Regresión Lineal

**Curso:** Aprendizaje Automático
**Profesor:** Ramón E. Álvarez S.
**Institución:** Instituto Tecnológico de las Américas (ITLA)

---

## 1. Introducción y Objetivo

El presente proyecto tiene como objetivo desarrollar un sistema de Machine Learning supervisado capaz de estimar el precio de cierre futuro de la acción de Meta Platforms (META), a partir de datos históricos del mercado bursátil obtenidos mediante la API de Alpaca Markets. Se entrenaron y compararon ocho modelos de regresión distintos, evaluando su desempeño mediante métricas estándar y analizando el impacto de sus respectivos hiperparámetros, con el fin de seleccionar el modelo con mejor capacidad predictiva.

---

## 2. Recolección y Preparación de los Datos

### 2.1 Fuente de datos

Los datos históricos se obtuvieron mediante la librería `alpaca-py`, utilizando velas diarias (`TimeFrame.Day`) del ticker **META**, con un rango desde el 1 de enero de 2018 hasta la fecha de corte del proyecto. Esto garantizó un dataset con más de 800 registros, cumpliendo el requisito mínimo establecido.

### 2.2 Variables originales obtenidas

`open`, `high`, `low`, `close`, `volume`, `trade_count`, `vwap`.

### 2.3 Ingeniería de características

A partir de las variables originales se construyeron nuevas variables (features) orientadas a capturar tendencia, momentum y volatilidad del activo:

| Variable        | Descripción                              | Justificación                                                                                                     |
| --------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `close`         | Precio de cierre                         | Punto de partida más directo para predecir el precio futuro; mayor correlación esperada con la variable objetivo. |
| `volume`        | Volumen de transacciones                 | Un volumen elevado suele anticipar movimientos de precio más significativos.                                      |
| `daily_return`  | Retorno diario (%)                       | Captura el momentum reciente del activo.                                                                          |
| `sma_7`         | Media móvil de 7 días                    | Suaviza el ruido diario y refleja la tendencia de corto plazo.                                                    |
| `volatility_7`  | Desviación estándar móvil de 7 días      | Mide la inestabilidad reciente del precio.                                                                        |
| `daily_range`   | Diferencia entre máximo y mínimo del día | Refleja la magnitud de la presión compradora/vendedora intradía.                                                  |
| `volume_change` | Variación porcentual del volumen         | Detecta cambios abruptos en el interés del mercado.                                                               |

**Variable objetivo (y):** `target_close_next`, correspondiente al precio de cierre del día siguiente, obtenido mediante `close.shift(-1)`.

### 2.4 Eliminación de variables redundantes

Durante el Análisis Exploratorio de Datos se identificó una correlación de 0.99 entre `sma_7` y `sma_21` (media móvil de 21 días), además de una correlación de 1.00 entre `sma_21` y `close`. Debido a esta redundancia, se decidió **eliminar la variable `sma_21`**, conservando `sma_7` como representante de la tendencia de corto plazo. Esta decisión se ve reforzada posteriormente por el modelo Lasso (ver sección 5.3), que confirmó de forma independiente la baja capacidad predictiva adicional de variables redundantes.

### 2.5 Limpieza y normalización

- Se eliminaron filas con valores nulos generados por el cálculo de medias móviles y desplazamientos temporales (`rolling`, `shift`).
- No fue necesaria codificación de variables categóricas, ya que el dataset es puramente numérico.
- Se aplicó **estandarización** (`StandardScaler` de scikit-learn) a las variables de entrada, transformándolas a media 0 y desviación estándar 1. El `scaler` ajustado se almacenó (`models/scaler.pkl`) para garantizar que los datos nuevos, al momento de la predicción, se transformen con los mismos parámetros estadísticos usados en el entrenamiento.

---

## 3. Análisis Exploratorio de Datos (EDA)

### 3.1 Serie de tiempo del precio de cierre

_(Insertar imagen: `docs/eda_serie_tiempo.png`)_

El precio de META mostró una tendencia general alcista durante el periodo analizado, con episodios de mayor volatilidad en tramos específicos, consistentes con eventos relevantes del mercado.

### 3.2 Distribución del retorno diario

_(Insertar imagen: `docs/eda_distribucion_retorno.png`)_

La distribución del retorno diario se centra alrededor de 0, indicando que en la mayoría de los días el precio no varía drásticamente respecto al día anterior. Sin embargo, se observan colas extendidas hacia ambos lados, reflejando días con movimientos atípicos asociados a reacciones del mercado ante noticias relevantes de la empresa.

### 3.3 Matriz de correlación entre variables

_(Insertar imagen: `docs/eda_correlacion.png`)_

Se identificó una fuerte multicolinealidad entre `close`, `sma_7` y `sma_21` (correlaciones de 0.99–1.00), esperable dado que las medias móviles se calculan directamente sobre el precio de cierre. Esto motivó la eliminación de `sma_21` (ver sección 2.4). Por otro lado, `daily_return` y `volume_change` mostraron correlación prácticamente nula con la variable objetivo, aunque se mantuvieron en el modelo por su relevancia teórica dentro del análisis técnico.

### 3.4 Volumen vs. Retorno diario

_(Insertar imagen: `docs/eda_volumen_vs_retorno.png`)_

Si bien la mayoría de las observaciones se concentran en valores bajos de ambas variables, los días con volumen extremadamente alto coinciden con los retornos más extremos (tanto positivos como negativos), sugiriendo que picos de volumen anticipan movimientos de precio significativos en cualquier dirección — una relación no lineal que el coeficiente de correlación no logra capturar completamente.

### 3.5 Distribución de la volatilidad (7 días)

_(Insertar imagen: `docs/eda_boxplot_volatilidad.png`)_

La distribución de la volatilidad presenta una marcada asimetría positiva: la mayoría de los días muestran volatilidad baja y estable, mientras que un número considerable de valores atípicos evidencia episodios puntuales de alta inestabilidad, consistente con el fenómeno de _"volatility clustering"_ característico de los mercados financieros.

---

## 4. Entrenamiento del Modelo

### 4.1 División de datos

El dataset se dividió en un conjunto de entrenamiento (80%) y un conjunto de prueba (20%), respetando estrictamente el orden cronológico de las observaciones (`shuffle=False`). Esta decisión es fundamental en problemas de series de tiempo financieras, ya que evita que el modelo "vea" información futura durante el entrenamiento.

### 4.2 Modelos entrenados

Se entrenaron y evaluaron los ocho modelos de regresión requeridos: Ordinary Least Squares, Ridge, Lasso, Bayesian Ridge, Nearest Neighbors (KNN), Random Forest, Support Vector Machine (SVM) y Neural Network (MLP).

---

## 5. Hiperparámetros y Análisis de su Impacto

### 5.1 Ridge Regression

Se evaluaron valores de `alpha` entre 0.01 y 100. El rendimiento disminuyó conforme aumentó la regularización, llegando a un colapso notable en `alpha=100` (R²=0.6479). El mejor resultado se obtuvo con **alpha=0.01** (R²=0.9470), prácticamente igual al de OLS sin regularización, lo que indica que la multicolinealidad detectada no perjudica significativamente al modelo lineal base.

### 5.2 Lasso Regression

Con **alpha=0.01**, Lasso obtuvo el mejor resultado global del proyecto (R²=0.9472). Su regularización L1 eliminó por completo el coeficiente de la variable `volatility_7` (0.000000), evidenciando que su información es redundante frente a `daily_range`, con la cual mantiene correlación considerable (0.60).

### 5.3 Bayesian Ridge Regression

No requiere ajuste manual de hiperparámetros, ya que estima automáticamente el nivel de regularización a partir de los datos. Obtuvo un resultado prácticamente idéntico a OLS (R²=0.9470), confirmando que el nivel óptimo de regularización para este problema es muy bajo.

### 5.4 K-Nearest Neighbors (KNN)

Se evaluaron valores de k entre 3 y 50. El rendimiento empeoró consistentemente al aumentar k (de R²=-2.86 con k=3 hasta R²=-4.38 con k=50). Se seleccionó **k=3** como configuración final, aunque el modelo resultó inadecuado para este problema debido a la no estacionariedad del precio de META: al tratarse de un activo con tendencia alcista sostenida, los "vecinos más cercanos" identificados pertenecen a periodos con niveles de precio considerablemente más bajos que los del conjunto de prueba, generando una subestimación sistemática.

### 5.5 Random Forest Regression

Se evaluaron valores de `max_depth` (3, 5, 10, None). El mejor resultado se obtuvo con **max_depth=5** (R²=-0.7955), aunque el desempeño general fue deficiente. Se confirmó que el precio máximo del conjunto de prueba ($790.00) superó considerablemente el máximo visto en entrenamiento ($595.94), y las predicciones del modelo quedaron efectivamente "topadas" cerca de este último ($589.77), evidenciando la incapacidad estructural de los modelos basados en árboles para extrapolar más allá del rango de valores observado.

### 5.6 Support Vector Machine (SVM)

Se evaluaron valores de `C` entre 0.1 y 50,000 con kernel RBF. El rendimiento mejoró sostenidamente hasta estabilizarse entre `C=5000` y `C=10000`, seleccionando **C=5000** (R²=-2.9467) como configuración final. A pesar de la optimización, el modelo mantuvo un desempeño deficiente, consistente con la misma limitación estructural observada en KNN y Random Forest.

### 5.7 Neural Network MLP Regression

Se evaluaron cinco arquitecturas, desde una capa de 10 neuronas hasta tres capas con 175 neuronas totales. La configuración de dos capas **(50, 25)** obtuvo el mejor resultado (R²=0.9309), superando incluso a arquitecturas más grandes. Esto sugiere que redes más profundas no aportan ventaja adicional en un problema con relación predominantemente lineal entre variables.

---

## 6. Métricas y Estadísticas de Rendimiento

| Modelo                      | MAE    | MSE      | RMSE   | R²         |
| --------------------------- | ------ | -------- | ------ | ---------- |
| **Lasso (α=0.01)**          | 10.23  | 219.10   | 14.80  | **0.9472** |
| OLS                         | 10.26  | 219.87   | 14.83  | 0.9470     |
| Bayesian Ridge              | 10.27  | 219.98   | 14.83  | 0.9470     |
| Ridge (α=0.01)              | 10.85  | 238.29   | 15.44  | 0.9426     |
| MLP (50, 25)                | 11.93  | 286.94   | 16.94  | 0.9309     |
| Random Forest (max_depth=5) | 66.98  | 7451.11  | 86.32  | -0.7955    |
| SVM (C=5000)                | 87.40  | 16378.93 | 127.98 | -2.9467    |
| KNN (k=3)                   | 114.44 | 16722.04 | 129.31 | -3.0294    |

_(Insertar imágenes: `docs/comparacion_r2.png` y `docs/comparacion_rmse.png`)_

### Selección del modelo final

De los ocho modelos evaluados, **Lasso Regression (alpha=0.01)** fue seleccionado como el modelo con mejor capacidad predictiva, al obtener el mejor resultado en las cuatro métricas simultáneamente. Le siguen muy de cerca OLS y Bayesian Ridge, con diferencias marginales entre los cuatro modelos lineales.

Esta consistencia se explica por la naturaleza predominantemente lineal de la relación entre las variables predictoras —particularmente `close` y `sma_7`— y la variable objetivo, evidenciada desde el análisis exploratorio (correlación de 0.99–1.00). La ventaja específica de Lasso radica en su capacidad de regularización L1, que simplificó el modelo eliminando `volatility_7` sin sacrificar capacidad predictiva.

En contraste, los modelos basados en distancia o particiones del espacio (KNN, Random Forest, SVM) presentaron un desempeño notablemente deficiente, producto de una limitación estructural común: la incapacidad de extrapolar más allá del rango de precios observado en entrenamiento, en un activo con tendencia alcista sostenida. El modelo MLP, si bien logró un desempeño aceptable, no logró superar a los modelos estrictamente lineales, reforzando la conclusión de que la complejidad adicional no aporta valor en un problema con una relación subyacente tan lineal.

---

## 7. Explicación del Funcionamiento del Sistema de Predicción

El sistema de predicción (`src/predict.py`) permite generar estimaciones del precio futuro de META a partir de un archivo `.csv` con datos históricos nuevos, siguiendo estos pasos:

1. **Carga de artefactos entrenados**: se cargan el modelo Lasso (`models/modelo_final.pkl`) y el `scaler` (`models/scaler.pkl`) generados durante la fase de entrenamiento, evitando la necesidad de reentrenar el sistema cada vez que se desea predecir.

2. **Preparación de los datos nuevos**: el CSV de entrada pasa por exactamente el mismo pipeline de ingeniería de características aplicado durante el entrenamiento (cálculo de `daily_return`, `sma_7`, `volatility_7`, `daily_range`, `volume_change`), garantizando consistencia entre los datos de entrenamiento y los datos de predicción. El sistema requiere un mínimo de 7 registros consecutivos, ya que las variables basadas en ventanas móviles necesitan ese historial mínimo para calcularse.

3. **Escalado consistente**: las variables se transforman utilizando el mismo `scaler` ajustado durante el entrenamiento (`scaler.transform()`, no `fit_transform()`), asegurando que los datos nuevos se normalicen bajo los mismos parámetros estadísticos aprendidos originalmente.

4. **Generación de la predicción**: el modelo Lasso entrenado predice el precio de cierre del día siguiente para cada fila del conjunto de datos nuevo.

5. **Exportación de resultados**: las predicciones se almacenan en `data/predicciones.csv`, junto con la fecha y el precio de cierre real correspondiente, facilitando su revisión.

El sistema se ejecuta mediante línea de comandos:

```bash
python src/predict.py --csv ruta/al/archivo_nuevo.csv
```

---

## 8. Conclusiones

El proyecto permitió comparar de forma rigurosa ocho enfoques distintos de regresión aplicados a la predicción de precios bursátiles, evidenciando que, para un activo con una relación predominantemente lineal entre sus variables técnicas y el precio futuro, los modelos lineales regularizados —particularmente Lasso Regression— ofrecen el mejor equilibrio entre precisión y simplicidad. Asimismo, el análisis reveló una limitación importante y recurrente en los modelos basados en distancia o particiones (KNN, Random Forest, SVM): su incapacidad de extrapolar en series de tiempo con tendencia sostenida, un hallazgo relevante para la selección de modelos en problemas financieros similares.

---

## Anexos

- Repositorio del proyecto: _(https://github.com/Abel-QL/Aprendizaje-Automatico-Regresion-Lineal)_
- Notebook completo: `notebooks/modelo.ipynb`
- Tablero Kanban de gestión: _(https://github.com/users/Abel-QL/projects/10)_
