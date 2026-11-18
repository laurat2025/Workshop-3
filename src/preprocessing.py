import re
import numpy as np
import pandas as pd

# Mapeo de estandarización de nombres de columnas entre los 5 años del World Happiness Report (2015-2019)
COLUMN_MAPPING = {
    "country": "country",
    "country or region": "country",
    "region": "region",
    "year": "year",
    "happiness score": "happiness_score",
    "happiness.score": "happiness_score",
    "score": "happiness_score",
    "economy (gdp per capita)": "gdp_per_capita",
    "economy..gdp.per.capita.": "gdp_per_capita",
    "gdp per capita": "gdp_per_capita",
    "family": "social_support",
    "social support": "social_support",
    "health (life expectancy)": "healthy_life_expectancy",
    "health..life.expectancy.": "healthy_life_expectancy",
    "healthy life expectancy": "healthy_life_expectancy",
    "freedom": "freedom",
    "freedom to make life choices": "freedom",
    "trust (government corruption)": "perceptions_of_corruption",
    "trust..government.corruption.": "perceptions_of_corruption",
    "perceptions of corruption": "perceptions_of_corruption",
    "generosity": "generosity",
}

FEATURE_COLUMNS = [
    "gdp_per_capita",
    "social_support",
    "healthy_life_expectancy",
    "freedom",
    "generosity",
    "perceptions_of_corruption",
]

TARGET_COLUMN = "happiness_score"


def normalize_column_name(col: str) -> str:
    clean = col.strip().lower()
    return COLUMN_MAPPING.get(clean, re.sub(r"[^a-z0-9]+", "_", clean).strip("_"))


def audit_yearly_schemas(df_raw: pd.DataFrame) -> dict:
    """
    Inspecciona los registros obtenidos por año, identificando columnas comunes,
    columnas exclusivas por año y conteo de valores faltantes antes de armonizar.
    """
    df = df_raw.drop(columns=[c for c in df_raw.columns if c.startswith("_kafka_")], errors="ignore").copy()
    year_col = next((c for c in df.columns if c.strip().lower() == "year"), None)

    audit = {
        "total_raw_rows": int(len(df)),
        "raw_columns": list(df.columns),
        "years_summary": {},
        "common_columns_across_years": [],
        "unique_columns_by_year": {},
    }

    if year_col is None:
        return audit

    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    years = sorted(df[year_col].dropna().unique().astype(int).tolist())
    active_cols_per_year = {}

    for yr in years:
        sub = df[df[year_col] == yr]
        non_empty_cols = {c for c in sub.columns if sub[c].notna().any()}
        active_cols_per_year[yr] = non_empty_cols
        audit["years_summary"][str(yr)] = {
            "row_count": int(len(sub)),
            "active_column_count": len(non_empty_cols),
            "null_counts": {c: int(sub[c].isna().sum()) for c in non_empty_cols if sub[c].isna().sum() > 0},
        }

    if active_cols_per_year:
        common = set.intersection(*active_cols_per_year.values())
        audit["common_columns_across_years"] = sorted(common)
        for yr, cols in active_cols_per_year.items():
            audit["unique_columns_by_year"][str(yr)] = sorted(cols - common)

    return audit


def harmonize_and_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza esquemas heterogéneos entre los distintos años del World Happiness Report (2015-2019),
    limpia tipos de datos, trata valores nulos y elimina duplicados.
    """
    df = df_raw.copy()

    meta_cols = [c for c in df.columns if c.startswith("_kafka_")]
    df = df.drop(columns=meta_cols, errors="ignore")

    df.columns = [normalize_column_name(c) for c in df.columns]
    df = df.T.groupby(level=0).first().T

    numeric_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[TARGET_COLUMN]).copy()

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    subset_keys = [c for c in ["country", "year"] if c in df.columns]
    if subset_keys:
        df = df.drop_duplicates(subset=subset_keys, keep="last")

    return df.reset_index(drop=True)
