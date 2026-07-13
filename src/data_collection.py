"""
data_collection.py
Obtención de datos históricos de META usando la API de Alpaca Markets.
Requiere: pip install alpaca-py pandas
"""

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime

API_KEY = "PKUVFYCCL5D3AGVIHP23NTDZIK"
API_SECRET = "HVLqPpjcha1qphdPBrzj95ikxdWLGF6eQCcgLkKms78t"

TICKER = "META"

client = StockHistoricalDataClient(API_KEY, API_SECRET)

request = StockBarsRequest(
    symbol_or_symbols=[TICKER],
    timeframe=TimeFrame.Day,
    start=datetime(2018, 1, 1),
    end=datetime(2026, 7, 1),
)

bars = client.get_stock_bars(request)
df = bars.df.reset_index()

print(f"Registros obtenidos: {len(df)}")
print(df.head())

# Guardar los datos en un archivo CSV
df.to_csv("DATA/raw/meta_historico.csv", index=False)
print("Guardado en DATA/raw/meta_historico.csv")
