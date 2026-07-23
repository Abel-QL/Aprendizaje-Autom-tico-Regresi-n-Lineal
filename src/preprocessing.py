"""
Abel Quezada
Preprocesamiento del dataset histórico de META para el modelo de regresión.
"""

import joblib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("DATA/raw/meta_historico.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Nulos antes de limpiar:\n{df.isnull().sum()}")
df = df.dropna().reset_index(drop=True)


df["daily_return"] = df["close"].pct_change()
df["sma_7"] = df["close"].rolling(window=7).mean()

df["volatility_7"] = df["close"].rolling(window=7).std()
df["daily_range"] = df["high"] - df["low"]

df["volume_change"] = df["volume"].pct_change()
df["target_close_next"] = df["close"].shift(-1)


df = df.dropna().reset_index(drop=True)


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

processed_df = X_scaled.copy()
processed_df["target_close_next"] = y.reset_index(drop=True)
processed_df.to_csv("DATA/processed/meta_procesado.csv", index=False)

print("Guardado en DATA/processed/meta_procesado.csv")


os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")
print("Scaler guardado en models/scaler.pkl")
