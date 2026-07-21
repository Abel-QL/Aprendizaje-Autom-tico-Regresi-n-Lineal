"""
Preprocesamiento del dataset histórico de META para el modelo de regresión.
Requiere: pip install pandas numpy scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Cargar el dataset crudo
df = pd.read_csv("DATA/raw/meta_historico.csv")

# Asegurar orden cronológico (importante: nunca mezclar el orden en series de tiempo)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# 2. Limpieza de datos nulos/erróneos
print(f"Nulos antes de limpiar:\n{df.isnull().sum()}")
df = df.dropna().reset_index(drop=True)


# Retorno diario (variación porcentual del precio de cierre)
df["daily_return"] = df["close"].pct_change()

# Medias móviles (tendencia de corto y mediano plazo)
df["sma_7"] = df["close"].rolling(window=7).mean()

# Volatilidad (desviación estándar móvil de 7 días)
df["volatility_7"] = df["close"].rolling(window=7).std()

# Rango diario (high - low), mide cuánto se movió el precio en el día
df["daily_range"] = df["high"] - df["low"]

# Volumen relativo (compara el volumen del día contra su media de 7 días)
df["volume_change"] = df["volume"].pct_change()

df["target_close_next"] = df["close"].shift(-1)


df = df.dropna().reset_index(drop=True)

# 6. Seleccionar features finales (mínimo 6 requeridas por el proyecto)

features = [
    "close",
    "volume",
    "daily_return",
    "sma_7",
    "volatility_7",
    "daily_range",
    "volume_change",
]
target = "target_close_next"

X = df[features]
y = df[target]

print(f"Total de registros después de limpieza: {len(df)}")
print(f"Features utilizadas ({len(features)}): {features}")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

# 8. Guardar dataset procesado
processed_df = X_scaled.copy()
processed_df["target_close_next"] = y.reset_index(drop=True)
processed_df.to_csv("DATA/processed/meta_procesado.csv", index=False)

print("Guardado en data/processed/meta_procesado.csv")
