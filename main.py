import json
import logging
import os
from src.consumer import consume_kafka_records
from src.database import save_predictions_to_db
from src.eda_visualizations import generate_eda_and_evaluation_plots
from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, audit_yearly_schemas, harmonize_and_clean
from src.train_model import compare_and_train_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    logger.info("=== PASO 1: Extracción desde Apache Kafka ===")
    df_raw = consume_kafka_records()

    if df_raw.empty:
        logger.error("No se obtuvieron datos desde Kafka. Revisa la conexión o el tópico.")
        return

    df_raw.to_csv("data/raw_kafka_records.csv", index=False)

    logger.info("=== PASO 2: Auditoría Multi-Año, Preparación y EDA ===")
    schema_audit = audit_yearly_schemas(df_raw)
    with open("reports/schema_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(schema_audit, f, indent=4, ensure_ascii=False)
    logger.info(f"Años detectados en el flujo: {list(schema_audit['years_summary'].keys())}")

    df_clean = harmonize_and_clean(df_raw)
    df_clean.to_csv("data/clean_happiness_dataset.csv", index=False)
    df_clean[FEATURE_COLUMNS + [TARGET_COLUMN]].describe().T.to_csv("reports/eda_descriptive_stats.csv")
    logger.info(f"Dataset limpio consolidado: {df_clean.shape[0]} filas, {df_clean.shape[1]} columnas.")

    logger.info("=== PASO 3: Entrenamiento y Comparación de Modelos ===")
    results = compare_and_train_models(df_clean, primary_model="linear")
    logger.info(f"Comparativa de modelos: {results['comparison']}")
    logger.info(f"Métricas del modelo seleccionado (Linear Regression): {results['metrics']}")

    logger.info("=== PASO 4: Generación de Gráficas y Carga a SQLite ===")
    plots = generate_eda_and_evaluation_plots(df_clean, results["y_test"], results["y_pred"])
    logger.info(f"Visualizaciones exportadas en: {plots}")

    df_eval = results["X_test"].copy()
    df_eval["actual_score"] = results["y_test"].values
    df_eval["predicted_score"] = results["y_pred"]
    df_eval["abs_error"] = (df_eval["actual_score"] - df_eval["predicted_score"]).abs()

    save_predictions_to_db(df_eval, table_name="happiness_predictions")
    logger.info("Pipeline completado exitosamente. Resultados guardados en SQLite.")


if __name__ == "__main__":
    run_pipeline()
