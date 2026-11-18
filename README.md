# Workshop 003: ETL Pipeline con Apache Kafka + Machine Learning

**Universidad Autónoma de Occidente (UAO)**  
**Facultad de Ingeniería y Ciencias Básicas — Programa: Ingeniería de Datos e Inteligencia Artificial**  
**Curso:** Extracción, Transformación y Carga (ETL - G01)  
**Estudiante:** Laura Vanessa Torres Valencia (`group.id`: `laura-vanessa-torres-valencia`)

---

## 1. Arquitectura del Sistema y Flujo de Datos (Diagrama de Bloques)

Este proyecto integra procesamiento de eventos en streaming mediante **Apache Kafka** con un pipeline de **Machine Learning supervisado (Regresión)** para predecir el puntaje de felicidad (*Happiness Score*) de distintos países a partir de indicadores socioeconómicos del *World Happiness Report* (2015–2019).

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTA & CONSTRUCCIÓN DE ESTADO                                          │
│    [Apache Kafka Broker: kafka-classroom-2026]                               │
│         └── Tópico: etl.records (SASL_SSL / SCRAM-SHA-512)                   │
│         └── Consumidor: group.id = "laura-vanessa-torres-valencia"            │
│         └── src/consumer.py ──► Acumulación en Memoria + Auditoría Partición │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. PREPARACIÓN DE DATOS, ARMONIZACIÓN MULTI-AÑO Y EDA                        │
│    [src/preprocessing.py]                                                    │
│         └── Normalización de encabezados heterogéneos (2015–2019)            │
│         └── Conversión numérica, tratamiento de nulos y deduplicación        │
│         └── Selección de 6 predictores socioeconómicos clave                 │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. ENTRENAMIENTO SIN DATA LEAKAGE Y EVALUACIÓN                               │
│    [src/train_model.py]                                                      │
│         └── Split Train/Test (80% / 20%, random_state=42)                    │
│         └── Pipeline Scikit-Learn: SimpleImputer(median) + StandardScaler    │
│         └── Comparación: Linear Regression vs Ridge vs Random Forest         │
│         └── Métricas: MAE, MSE, RMSE, R² y R² Ajustado                       │
└─────────────────────────┬────────────────────────────────────┬───────────────┘
                          │                                    │
                          ▼                                    ▼
┌──────────────────────────────────────┐ ┌─────────────────────────────────────┐
│ 4A. PERSISTENCIA Y AUDITORÍA SQL     │ │ 4B. INFERENCIA EN STREAMING EN VIVO │
│  [models/happiness_regression_model] │ │  [src/streaming_inference.py]       │
│  [src/database.py] ──► SQLite DB     │ │  [src/dashboard.py] (Streamlit)     │
└──────────────────────────────────────┘ └─────────────────────────────────────┘
```

---

## 2. Decisiones Técnicas, Validación y Metodología

### 2.1. Extracción y Construcción de Estado en Memoria (`src/consumer.py`)
* **Conexión segura:** Autenticación mediante protocolo `SASL_SSL` y mecanismo `SCRAM-SHA-512` empleando certificados raíz de `certifi.where()`.
* **Identificador único de consumidor:** Configurado explícitamente como `group.id = "laura-vanessa-torres-valencia"` para aislar el seguimiento de *offsets* respecto a otros estudiantes del curso.
* **Control de *offsets* y verificación:** Se desactiva el *auto-commit* (`enable.auto.commit = False`) para confirmar cada mensaje únicamente tras decodificar su payload JSON, capturar errores de formato y extraer sus metadatos (`_kafka_topic`, `_kafka_partition`, `_kafka_offset`, `_kafka_timestamp`). Al finalizar el consumo, se registra el conteo total de mensajes por partición.

### 2.2. Integración Multi-Año (2015–2019), Justificación y Limitaciones (`src/preprocessing.py`)
* **Origen de los datos:** Los registros de los 5 archivos anuales ya residen publicados en el tópico `etl.records`, por lo que no se requiere descarga manual de CSVs externos.
* **Lógica de unificación (*Merge / Harmonization*):** Dado que los reportes anuales utilizan variantes sintácticas en sus columnas (por ejemplo, `Happiness Score`, `Happiness.Score` y `Score`; o `Family` y `Social support`), se implementó una tabla de equivalencia semántica (`COLUMN_MAPPING`) que proyecta los 5 años hacia un esquema canónico único.
* **Justificación:** Apilar los 5 periodos aumenta el tamaño muestral (~780 observaciones) y aporta mayor diversidad económica y geográfica, mejorando la estabilidad de los coeficientes de regresión.
* **Limitaciones conocidas:** Al consolidar datos longitudinales en un modelo transversal agrupado (*pooled OLS*), las observaciones de un mismo país en años consecutivos presentan autocorrelación temporal y no son estrictamente independientes entre sí.

### 2.3. Selección de Variables (*Feature Selection*)
* **Variables predictoras (`X`):** `gdp_per_capita`, `social_support`, `healthy_life_expectancy`, `freedom`, `generosity` y `perceptions_of_corruption`.
* **Variable objetivo (`y`):** `happiness_score`.
* **Criterio de exclusión:** Se descartaron variables nominales (`country`, `region`) y columnas derivadas directamente de la suma del puntaje (`happiness_rank`, `whisker_high`, `whisker_low`, `dystopia_residual`) para evitar fuga de información (*target leakage*).

### 2.4. Prevención de *Data Leakage* y Comparación de Modelos (`src/train_model.py`)
* **Encapsulamiento en `Pipeline`:** Tanto la imputación de valores faltantes (`SimpleImputer(strategy="median")`) como el escalamiento (`StandardScaler()`) están encapsulados dentro de un `Pipeline` de Scikit-Learn. De este modo, los parámetros de transformación (medianas, medias y desviaciones estándar) se calculan **exclusivamente sobre `X_train`** (80%) y luego se aplican sobre `X_test` (20%).
* **Evaluación comparativa:** Se evalúan `LinearRegression`, `Ridge(alpha=1.0)` y `RandomForestRegressor(n_estimators=200)` reportando **MAE, MSE, RMSE, $R^2$ y $R^2_{ajustado}$** en `models/metrics_report.json`.

---

## 3. Estructura del Repositorio

```text
workshop-003-kafka-ml/
├── .env                                  # Variables de entorno configuradas (group.id y sasl.password)
├── .env.example                          # Plantilla de variables de entorno
├── .gitignore                            # Exclusión de entornos virtuales y archivos temporales
├── requirements.txt                      # Dependencias del proyecto
├── README.md                             # Documentación técnica completa
├── main.py                               # Orquestador principal de extremo a extremo
├── notebooks/
│   └── Workshop_003_Kafka_ML.ipynb       # Notebook autocontenido (listo para Google Colab o Jupyter)
├── src/
│   ├── __init__.py
│   ├── config.py                         # Configuración centralizada de Kafka, DB y rutas
│   ├── consumer.py                       # Consumidor desde etl.records + verificación de estado
│   ├── preprocessing.py                  # Armonización de esquemas (2015-2019) y limpieza
│   ├── train_model.py                    # Comparación de modelos, validación sin data leakage y métricas
│   ├── database.py                       # Persistencia en SQLite
│   ├── streaming_inference.py            # Consumidor continuo de inferencia en tiempo real
│   └── dashboard.py                      # Visualización interactiva con Streamlit
├── sql/
│   └── queries.sql                       # Consultas SQL de auditoría y métricas
├── data/                                 # Carpeta destino para CSVs y base de datos SQLite
└── models/                               # Carpeta destino para artefactos .joblib y reportes JSON
```

---


