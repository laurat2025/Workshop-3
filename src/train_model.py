import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_PATH
from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN


def build_training_pipeline(model_type: str = "linear") -> Pipeline:
    """
    Construye un Pipeline de Scikit-Learn que encadena imputación + escalamiento
    dentro del flujo de entrenamiento para evitar fuga de datos (Data Leakage).
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, FEATURE_COLUMNS),
        ]
    )

    if model_type == "ridge":
        regressor = Ridge(alpha=1.0)
    elif model_type == "rf":
        regressor = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        regressor = LinearRegression()

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def compare_and_train_models(df: pd.DataFrame, primary_model: str = "linear") -> dict:
    """
    Divide los datos en train/test (80/20), compara Linear Regression, Ridge y Random Forest
    sin data leakage, calcula MAE, MSE, RMSE, R2 y R2 ajustado, y serializa el modelo seleccionado.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    candidates = ["linear", "ridge", "rf"]
    comparison_report = {}
    trained_artifacts = {}

    for m_type in candidates:
        pipeline = build_training_pipeline(model_type=m_type)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, y_pred)

        n = len(y_test)
        p = len(FEATURE_COLUMNS)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

        comparison_report[m_type] = {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "MAE": round(float(mae), 4),
            "MSE": round(float(mse), 4),
            "RMSE": round(float(rmse), 4),
            "R2": round(float(r2), 4),
            "R2_adjusted": round(float(r2_adj), 4),
        }
        trained_artifacts[m_type] = (pipeline, y_pred)

    selected_pipeline, selected_pred = trained_artifacts[primary_model]
    joblib.dump(selected_pipeline, MODEL_PATH)

    full_report = {
        "selected_model": primary_model,
        "comparison": comparison_report,
    }
    with open("models/metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4)

    return {
        "pipeline": selected_pipeline,
        "metrics": comparison_report[primary_model],
        "comparison": comparison_report,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": selected_pred,
    }
