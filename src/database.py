import sqlite3
import pandas as pd
from src.config import DB_PATH


def save_predictions_to_db(df_results: pd.DataFrame, table_name: str = "happiness_predictions") -> None:
    """
    Persiste los registros procesados y sus predicciones en una base de datos SQLite.
    """
    with sqlite3.connect(DB_PATH) as conn:
        df_results.to_sql(table_name, conn, if_exists="replace", index=False)


def load_predictions_from_db(table_name: str = "happiness_predictions") -> pd.DataFrame:
    """
    Consulta las predicciones almacenadas en SQLite.
    """
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)
