import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN

sns.set_theme(style="whitegrid")


def generate_eda_and_evaluation_plots(
    df_clean: pd.DataFrame,
    y_test: pd.Series,
    y_pred,
    output_dir: str = "reports",
) -> list[str]:
    """
    Genera y guarda en disco las gráficas de EDA (distribución, correlación, outliers)
    y de evaluación del modelo (Real vs Predicho y Residuales) al ejecutar main.py.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    # 1. Distribución del Target + Matriz de Correlación
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(df_clean[TARGET_COLUMN], kde=True, color="#1f77b4", ax=axes[0])
    axes[0].set_title("Distribución del Happiness Score (2015–2019)")
    axes[0].set_xlabel("Happiness Score")

    corr = df_clean[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=axes[1])
    axes[1].set_title("Matriz de Correlación de Pearson")
    plt.tight_layout()
    p1 = os.path.join(output_dir, "01_eda_distribucion_correlacion.png")
    plt.savefig(p1, dpi=150)
    plt.close(fig)
    saved_paths.append(p1)

    # 2. Boxplots de Outliers
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.boxplot(data=df_clean[FEATURE_COLUMNS], orient="h", palette="Blues", ax=ax)
    ax.set_title("Distribución y Valores Atípicos (Outliers) por Predictor")
    plt.tight_layout()
    p2 = os.path.join(output_dir, "02_eda_outliers_boxplot.png")
    plt.savefig(p2, dpi=150)
    plt.close(fig)
    saved_paths.append(p2)

    # 3. Evaluación del Modelo: Real vs Predicho y Residuales
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test, y_pred, alpha=0.7, color="#2b5c8f")
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal (y = x)")
    axes[0].set_title("Valores Reales vs. Predicciones")
    axes[0].set_xlabel("Happiness Score Real")
    axes[0].set_ylabel("Happiness Score Predicho")
    axes[0].legend()

    sns.histplot(residuals, kde=True, color="#2b5c8f", ax=axes[1])
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title("Distribución de Residuales")
    axes[1].set_xlabel("Residual")
    plt.tight_layout()
    p3 = os.path.join(output_dir, "03_model_evaluation_residuals.png")
    plt.savefig(p3, dpi=150)
    plt.close(fig)
    saved_paths.append(p3)

    return saved_paths
