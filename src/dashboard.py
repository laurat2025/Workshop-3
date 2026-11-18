import sqlite3
import pandas as pd
import streamlit as st
from src.config import DB_PATH

st.set_page_config(page_title="World Happiness - Kafka & ML Monitor", layout="wide")
st.title("Monitor en Tiempo Real & Evaluación Predictiva — World Happiness")

try:
    with sqlite3.connect(DB_PATH) as conn:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()

        if "happiness_predictions" in tables:
            st.subheader("1. Evaluación del Modelo de Regresión (Set de Prueba)")
            df_eval = pd.read_sql("SELECT * FROM happiness_predictions", conn)
            col1, col2, col3 = st.columns(3)
            col1.metric("Registros Evaluados", len(df_eval))
            col2.metric("MAE Promedio", f"{df_eval['abs_error'].mean():.4f}")
            col3.metric("Score Promedio Real", f"{df_eval['actual_score'].mean():.3f}")

            st.scatter_chart(df_eval, x="actual_score", y="predicted_score")
            st.dataframe(df_eval.head(20), use_container_width=True)

        if "streaming_inference_log" in tables:
            st.subheader("2. Log Operacional de Inferencia en Streaming (Kafka)")
            df_stream = pd.read_sql("SELECT * FROM streaming_inference_log ORDER BY id DESC LIMIT 100", conn)
            st.metric("Eventos Procesados en Vivo", len(df_stream))
            st.dataframe(df_stream, use_container_width=True)

        if not tables:
            st.info("Aún no hay tablas en SQLite. Ejecuta primero el pipeline principal (`python main.py`).")
except Exception as e:
    st.error(f"Error leyendo la base de datos: {e}")
