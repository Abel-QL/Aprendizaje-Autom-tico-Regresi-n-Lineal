"""
Módulo de predicción: carga un .csv con nuevos datos históricos de META,
aplica el mismo pipeline de preprocesamiento y genera la predicción del
precio de cierre del día siguiente usando el modelo Lasso ya entrenado.
"""

import argparse
import pandas as pd
import numpy as np
import joblib

FEATURES = [
    "close",
    "volume",
    "daily_return",
    "sma_7",
    "volatility_7",
    "daily_range",
    "volume_change",
]


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["daily_return"] = df["close"].pct_change()
    df["sma_7"] = df["close"].rolling(window=7).mean()
    df["volatility_7"] = df["close"].rolling(window=7).std()
    df["daily_range"] = df["high"] - df["low"]
    df["volume_change"] = df["volume"].pct_change()

    df = df.dropna().reset_index(drop=True)

    return df


def predecir(csv_path: str):
    modelo = joblib.load("models/modelo_final.pkl")
    scaler = joblib.load("models/scaler.pkl")

    df_nuevo = pd.read_csv(csv_path)
    df_procesado = preparar_datos(df_nuevo)

    if df_procesado.empty:
        raise ValueError(
            "El CSV no tiene suficientes filas para calcular sma_7/volatility_7 "
            "(se necesitan al menos 7 registros consecutivos)."
        )

    X_nuevo = df_procesado[FEATURES]

    X_nuevo_scaled = scaler.transform(X_nuevo)

    predicciones = modelo.predict(X_nuevo_scaled)

    df_resultado = df_procesado[["timestamp", "close"]].copy()
    df_resultado["prediccion_precio_siguiente"] = predicciones

    return df_resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicción de precio futuro de META")
    parser.add_argument("--csv", required=True, help="Ruta al CSV con nuevos datos")
    args = parser.parse_args()

    resultado = predecir(args.csv)
    print(resultado)

    resultado.to_csv("DATA/predicciones.csv", index=False)
    print("\nPredicciones guardadas en DATA/predicciones.csv")
