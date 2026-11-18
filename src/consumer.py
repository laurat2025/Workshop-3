import json
import logging
from collections import Counter
import pandas as pd
from confluent_kafka import Consumer, KafkaException
from src.config import KAFKA_CONFIG, KAFKA_TOPIC

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def consume_kafka_records(max_empty_polls: int = 5, poll_timeout: float = 2.0) -> pd.DataFrame:
    """
    Se conecta al broker de Apache Kafka mediante SASL_SSL / SCRAM-SHA-512,
    consume todos los eventos disponibles en el tópico `etl.records`,
    mantiene los registros acumulados en un estado en memoria (lista -> DataFrame)
    y ejecuta una verificación formal de integridad y conteo por partición.
    """
    consumer = Consumer(KAFKA_CONFIG)
    consumer.subscribe([KAFKA_TOPIC])
    logger.info(f"Suscrito al tópico '{KAFKA_TOPIC}' con group.id='{KAFKA_CONFIG['group.id']}'")

    in_memory_state = []
    partition_counts = Counter()
    empty_polls = 0
    malformed_messages = 0

    try:
        while empty_polls < max_empty_polls:
            msg = consumer.poll(timeout=poll_timeout)

            if msg is None:
                empty_polls += 1
                logger.info(f"Sin mensajes nuevos ({empty_polls}/{max_empty_polls})...")
                continue

            if msg.error():
                raise KafkaException(msg.error())

            empty_polls = 0
            try:
                raw_value = msg.value().decode("utf-8")
                payload = json.loads(raw_value)
            except (UnicodeDecodeError, json.JSONDecodeError) as decode_err:
                malformed_messages += 1
                logger.warning(f"Mensaje malformado en offset {msg.offset()}: {decode_err}")
                consumer.commit(message=msg, asynchronous=False)
                continue

            # Extracción explícita de metadatos + campos originales
            record = {
                "_kafka_topic": msg.topic(),
                "_kafka_partition": msg.partition(),
                "_kafka_offset": msg.offset(),
                "_kafka_timestamp": msg.timestamp()[1],
            }
            if isinstance(payload, dict):
                record.update(payload)
            else:
                record["raw_payload"] = payload

            in_memory_state.append(record)
            partition_counts[msg.partition()] += 1
            consumer.commit(message=msg, asynchronous=False)

    except KafkaException as ke:
        logger.error(f"Error de comunicación con el broker de Kafka: {ke}")
        raise
    except KeyboardInterrupt:
        logger.info("Consumo detenido manualmente por el usuario.")
    finally:
        consumer.close()
        logger.info("=== VERIFICACIÓN DE ESTADO EN MEMORIA (STATE CONSTRUCTION) ===")
        logger.info(f"Total de registros acumulados: {len(in_memory_state)}")
        logger.info(f"Distribución por partición: {dict(partition_counts)}")
        logger.info(f"Mensajes descartados por error de formato: {malformed_messages}")

    df = pd.DataFrame(in_memory_state)
    return df


if __name__ == "__main__":
    df_records = consume_kafka_records()
    if not df_records.empty:
        print("\nColumnas detectadas:", df_records.columns.tolist())
        print(df_records.head())
        df_records.to_csv("data/raw_kafka_records.csv", index=False)
        print("Registros guardados en data/raw_kafka_records.csv")
    else:
        print("No se recibieron mensajes desde el tópico.")
