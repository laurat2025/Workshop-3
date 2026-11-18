import os
import certifi
from dotenv import load_dotenv

load_dotenv()

KAFKA_PASSWORD = os.getenv("KAFKA_SASL_PASSWORD")
if not KAFKA_PASSWORD:
    raise EnvironmentError(
        "Falta la variable de entorno KAFKA_SASL_PASSWORD. "
        "Configúrala en tu archivo .env (puedes guiarte con .env.example)."
    )

KAFKA_CONFIG = {
    "bootstrap.servers": os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        "kafka-classroom-2026.westus.cloudapp.azure.com:9092",
    ),
    "security.protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL"),
    "sasl.mechanism": os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-512"),
    "sasl.username": os.getenv("KAFKA_SASL_USERNAME", "classroom-students"),
    "sasl.password": KAFKA_PASSWORD,
    "ssl.ca.location": certifi.where(),
    "group.id": os.getenv("KAFKA_GROUP_ID", "laura-vanessa-torres-valencia"),
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,
}

KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "etl.records")
DB_PATH = os.getenv("DB_PATH", "data/happiness_predictions.db")
MODEL_PATH = os.getenv("MODEL_PATH", "models/happiness_regression_model.joblib")
