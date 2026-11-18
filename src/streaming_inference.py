import json
import logging
import sqlite3
from datetime import datetime, timezone
import joblib
import pandas as pd
from confluent_kafka import Consumer, KafkaException

from src.config import DB_PATH, KAFKA_CONFIG, KAFKA_TOPIC, MODEL_PATH
from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, harmonize_and_clean

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def init_streaming_table():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS streaming_inference_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                processed_at TEXT,
                kafka_partition INTEGER,
                kafka_offset INTEGER,
                country TEXT,
                year INTEGER,
                actual_score REAL,
                predicted_score REAL,
                abs_error REAL
            )
            """
        )


def run_realtime_inference_consumer():
    """
    Consumidor de aplicación continuo:
    Lee cada evento en tiempo real desde Kafka, normaliza el esquema,
    ejecuta inferencia con el modelo serializado (.joblib) y persiste en SQLite.
    """
    init_streaming_table()
    model = joblib.load(MODEL_PATH)
    logger.info(f"Modelo cargado exitosamente desde {MODEL_PATH}")

    stream_config = KAFKA_CONFIG.copy()
    stream_config["group.id"] = f"{KAFKA_CONFIG['group.id']}-realtime-inference"

    consumer = Consumer(stream_config)
    consumer.subscribe([KAFKA_TOPIC])
    logger.info(f"Escuchando eventos en vivo en el tópico '{KAFKA_TOPIC}' con group.id='{stream_config['group.id']}'...")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())

            payload = json.loads(msg.value().decode("utf-8"))
            df_event = harmonize_and_clean(pd.DataFrame([payload]))

            if df_event.empty:
                consumer.commit(message=msg, asynchronous=False)
                continue

            X_event = df_event[FEATURE_COLUMNS]
            pred_score = float(model.predict(X_event)[0])
            actual_score = float(df_event[TARGET_COLUMN].iloc[0])
            abs_error = abs(actual_score - pred_score)

            country = df_event["country"].iloc[0] if "country" in df_event.columns else "Unknown"
            year = int(df_event["year"].iloc[0]) if "year" in df_event.columns and pd.notna(df_event["year"].iloc[0]) else None

            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    """
                    INSERT INTO streaming_inference_log
                    (processed_at, kafka_partition, kafka_offset, country, year, actual_score, predicted_score, abs_error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(timezone.utc).isoformat(),
                        msg.partition(),
                        msg.offset(),
                        country,
                        year,
                        actual_score,
                        round(pred_score, 4),
                        round(abs_error, 4),
                    ),
                )

            logger.info(
                f"[Offset {msg.offset()}] País: {country} | Real: {actual_score:.3f} | "
                f"Predicción: {pred_score:.3f} | Error Abs: {abs_error:.3f}"
            )
            consumer.commit(message=msg, asynchronous=False)

    except KeyboardInterrupt:
        logger.info("Inferencia en tiempo real detenida.")
    finally:
        consumer.close()


if __name__ == "__main__":
    run_realtime_inference_consumer()
