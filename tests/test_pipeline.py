import numpy as np
import pandas as pd
from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, audit_yearly_schemas, harmonize_and_clean
from src.train_model import compare_and_train_models


def _build_mock_kafka_records() -> pd.DataFrame:
    """Simula registros heterogéneos provenientes de los 5 años (2015-2019) en Kafka."""
    rng = np.random.default_rng(42)
    rows = []
    # Esquema estilo 2015
    for i in range(25):
        gdp = rng.uniform(0.3, 1.6)
        fam = rng.uniform(0.4, 1.5)
        health = rng.uniform(0.2, 1.0)
        free = rng.uniform(0.1, 0.7)
        gen = rng.uniform(0.0, 0.6)
        trust = rng.uniform(0.0, 0.5)
        score = 2.0 + 1.5 * gdp + 1.2 * fam + 0.9 * health + 0.6 * free + rng.normal(0, 0.15)
        rows.append({
            "_kafka_topic": "etl.records",
            "_kafka_partition": 0,
            "_kafka_offset": i,
            "Country": f"Country_{i}",
            "Region": "Western Europe",
            "Year": 2015,
            "Happiness Score": round(float(score), 3),
            "Economy (GDP per Capita)": round(float(gdp), 3),
            "Family": round(float(fam), 3),
            "Health (Life Expectancy)": round(float(health), 3),
            "Freedom": round(float(free), 3),
            "Generosity": round(float(gen), 3),
            "Trust (Government Corruption)": round(float(trust), 3),
        })
    # Esquema estilo 2019
    for i in range(25):
        gdp = rng.uniform(0.3, 1.6)
        soc = rng.uniform(0.4, 1.5)
        health = rng.uniform(0.2, 1.0)
        free = rng.uniform(0.1, 0.7)
        gen = rng.uniform(0.0, 0.6)
        corr = rng.uniform(0.0, 0.5) if i % 7 != 0 else None  # incluye nulo intencional
        score = 2.0 + 1.5 * gdp + 1.2 * soc + 0.9 * health + 0.6 * free + rng.normal(0, 0.15)
        rows.append({
            "_kafka_topic": "etl.records",
            "_kafka_partition": 1,
            "_kafka_offset": i,
            "Country or region": f"Country_{i}",
            "Year": 2019,
            "Score": round(float(score), 3),
            "GDP per capita": round(float(gdp), 3),
            "Social support": round(float(soc), 3),
            "Healthy life expectancy": round(float(health), 3),
            "Freedom to make life choices": round(float(free), 3),
            "Generosity": round(float(gen), 3),
            "Perceptions of corruption": round(float(corr), 3) if corr is not None else None,
        })
    return pd.DataFrame(rows)


def test_schema_audit_and_harmonization():
    df_raw = _build_mock_kafka_records()
    audit = audit_yearly_schemas(df_raw)
    assert set(audit["years_summary"].keys()) == {"2015", "2019"}
    assert "Year" in audit["common_columns_across_years"]

    df_clean = harmonize_and_clean(df_raw)
    assert len(df_clean) == 50
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert col in df_clean.columns


def test_model_training_without_leakage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    df_raw = _build_mock_kafka_records()
    df_clean = harmonize_and_clean(df_raw)

    results = compare_and_train_models(df_clean, primary_model="linear")
    assert results["metrics"]["R2"] > 0.75
    assert (tmp_path / "models/metrics_report.json").exists()
