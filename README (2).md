# Workshop 003: ETL Pipeline con Apache Kafka + Machine Learning

**Universidad Autónoma de Occidente (UAO)**  
**Facultad de Ingeniería y Ciencias Básicas** — Programa: *Ingeniería de Datos e Inteligencia Artificial*  
**Curso:** ETL (G01)  
**Autora:** Laura Vanessa Torres Valencia  

---

## 1. Descripción General y Arquitectura

Este proyecto implementa un sistema predictivo integral que combina **Data Streaming con Apache Kafka** y **Machine Learning (Regresión)** para estimar el índice de felicidad mundial (*Happiness Score*) a partir de los registros históricos del *World Happiness Report* (2015–2019).

### Flujo de Trabajo (Workflow)

El pipeline sigue una arquitectura modular en cuatro fases secuenciales orquestadas desde `main.py` y replicadas de forma autocontenida en `notebooks/Workshop_003_Kafka_ML.ipynb`:

```text
[Topic Kafka: etl.records] 
       │
       ▼ (1. Ingesta en Streaming y Estado en Memoria)
[Consumidor Kafka (src/consumer.py)] 
       │
       ▼ (2. Preprocesamiento, Armonización y Limpieza)
[Data Preparation & EDA (src/preprocessing.py)] 
       │
       ├──► (3. Entrenamiento sin Data Leakage) ──► [Modelo (.joblib)]
       │
       ▼ (4. Inferencia Continua y Persistencia)
[Predicción + Carga a BD (src/database.py)] ──► [Dashboard & Auditoría (Streamlit & SQL)]
```

### Stack Tecnológico
* **Lenguaje:** Python 3.10+
* **Streaming:** Apache Kafka (`confluent-kafka`) con seguridad `SASL_SSL` / `SCRAM-SHA-512`
* **Procesamiento y Análisis:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Base de Datos:** SQLite (`sqlite3`)
* **Visualización:** `matplotlib`, `seaborn`, `streamlit`

---

## 2. Estructura del Repositorio

El repositorio está organizado siguiendo las mejores prácticas de ingeniería de software y versionamiento (excluyendo datos crudos, modelos binarios pesados y credenciales reales mediante `.gitignore`):

```text
workshop-003-kafka-ml/
├── .env.example              # Plantilla segura de variables de entorno requeridas
├── .gitignore                # Exclusión de .env, data/, models/, entornos y caché
├── requirements.txt          # Dependencias del proyecto
├── README.md                 # Documentación técnica completa
├── main.py                   # Orquestador principal del pipeline ETL + ML
├── notebooks/
│   └── Workshop_003_Kafka_ML.ipynb  # Notebook interactivo con evidencia del proceso y EDA
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuración centralizada de Kafka, rutas y BD
│   ├── consumer.py           # Consumidor Kafka y construcción de estado en memoria
│   ├── preprocessing.py      # Armonización de esquemas (2015–2019) y limpieza
│   ├── train_model.py        # División Train/Test, pipelines sin data leakage y entrenamiento
│   ├── database.py           # Persistencia de registros y predicciones en SQLite
│   ├── streaming_inference.py# Consumidor continuo para inferencia en tiempo real
│   └── dashboard.py          # Panel de visualización interactiva con Streamlit
├── sql/
│   └── queries.sql           # Consultas SQL de auditoría y métricas de negocio
└── tests/                    # Pruebas unitarias de validación de datos y pipeline
```

---

## 3. Documentación Técnica (Technical Documentation)

Esta sección detalla la implementación, validaciones, lógica de integración y decisiones de diseño adoptadas en el desarrollo del sistema.

### 3.1. Implementación y Validación (`consumer.py` & `tests/`)
* **Conexión Segura y Consumo:** El consumidor utiliza `confluent-kafka` conectado al clúster `kafka-classroom-2026.westus.cloudapp.azure.com:9092` mediante certificados SSL (`certifi.where()`), autenticación `SCRAM-SHA-512`, `auto.offset.reset = earliest` y el identificador único `group.id = "laura-vanessa-torres-valencia"` para leer desde el inicio del tópico `etl.records` sin interferir con otros consumidores.
* **Construcción de Estado en Memoria:** Los mensajes JSON decodificados se acumulan dinámicamente en una estructura en memoria (lista de diccionarios convertida posteriormente a `DataFrame`), detectando el fin de partición (`EOF`) para asegurar la ingesta completa.
* **Validación de Datos y Estado:**
  * Inspección de metadatos de Kafka (`topic`, `partition`, `offset`, `timestamp`) para descartar mensajes vacíos o malformados antes de la deserialización.
  * Auditoría de conteo de registros consumidos por año y verificación de integridad tras la carga en SQLite mediante las consultas en `sql/queries.sql`.
  * Pruebas unitarias en `tests/` para validar el correcto mapeo de columnas y la ausencia de valores nulos críticos.

### 3.2. Lógica de Fusión y Transformaciones (`preprocessing.py`)
Los datasets históricos del *World Happiness Report* (2015–2019) presentan diferencias estructurales en la nomenclatura y disponibilidad de columnas según el año:

| Concepto / Variable | Variaciones detectadas por año | Columna Canónica Estandarizada |
| :--- | :--- | :--- |
| **País** | `Country`, `Country or region` | `country` |
| **Variable Objetivo** | `Happiness Score`, `Happiness.Score`, `Score` | `happiness_score` |
| **PIB per cápita** | `Economy (GDP per Capita)`, `Economy..GDP.per.Capita.`, `GDP per capita` | `gdp_per_capita` |
| **Soporte Social** | `Family`, `Social support` | `social_support` |
| **Expectativa de Vida** | `Health (Life Expectancy)`, `Health..Life.Expectancy.`, `Healthy life expectancy` | `healthy_life_expectancy` |
| **Libertad** | `Freedom`, `Freedom to make life choices` | `freedom` |
| **Generosidad** | `Generosity` | `generosity` |
| **Corrupción** | `Trust (Government Corruption)`, `Perceptions of corruption` | `perceptions_of_corruption` |

* **Estrategia de Integración (Merge Logic):**
  Se optó por una **concatenación vertical (`pd.concat`) basada en alineación de esquema común**, conservando la dimensión temporal `year`, en lugar de un cruce horizontal (`JOIN` por país), dado que cada registro constituye una observación independiente País-Año.
  * *Justificación:* Maximiza el tamaño muestral disponible para entrenamiento y permite modelar las relaciones estructurales transversales entre indicadores socioeconómicos y bienestar.
  * *Limitaciones:* Se descartaron columnas presentes únicamente en años específicos (como intervalos de confianza o *Dystopia Residual*) para evitar alta dispersión de valores nulos, asumiendo estacionariedad en las escalas de contribución entre periodos.
* **Transformaciones y Prevención de Data Leakage:**
  * La partición en entrenamiento (80%) y prueba (20%) se ejecuta **antes** de cualquier cálculo estadístico.
  * La imputación de valores faltantes (`SimpleImputer` con estrategia de mediana) y el escalado (`StandardScaler`) están encapsulados en un `Pipeline` de `scikit-learn` ajustado (`fit`) exclusivamente sobre `X_train` y aplicado (`transform`) sobre `X_test`.

### 3.3. Decisiones Relevantes de Diseño (`train_model.py` & `database.py`)
1. **Selección de Características (Feature Selection):** A partir del EDA y las matrices de correlación, se seleccionaron las 6 variables predictoras con mayor capacidad explicativa (`gdp_per_capita`, `social_support`, `healthy_life_expectancy`, `freedom`, `generosity`, `perceptions_of_corruption`). Se excluyeron `country` y `happiness_rank` para evitar *overfitting* y fuga directa del target.
2. **Evaluación de Modelos de Regresión:** Se utilizó **Linear Regression** como modelo base (*baseline*) por su interpretabilidad directa sobre factores aditivos, comparándolo frente a modelos regularizados y de ensamble (`Ridge`, `RandomForestRegressor`) mediante las métricas **MAE**, **MSE**, **RMSE** y **$R^2$**, además de gráficos de valores reales vs. predichos y análisis de residuales.
3. **Persistencia con SQLite:** Se eligió SQLite (`data/happiness_predictions.db`) por ser una base de datos relacional embebida, ligera y sin dependencias de servidor externo, garantizando total portabilidad entre ejecución local y Google Colab.
4. **Seguridad de Credenciales:** Las claves del broker Kafka se gestionan mediante variables de entorno (`.env` bloqueado en `.gitignore`), dejando `.env.example` como plantilla pública segura.

---

## 4. Guía de Ejecución y Reproducción

### Opción A: Ejecución en Google Colab (Notebook Interactivo)
1. Abre o sube el archivo `notebooks/Workshop_003_Kafka_ML.ipynb` en Google Colab.
2. En la celda de configuración de Kafka, verifica o ingresa tu `group.id` único y la contraseña (`sasl.password`) del curso.
3. Ejecuta todas las celdas secuencialmente para reproducir el consumo desde Kafka, el EDA, el entrenamiento del modelo y las visualizaciones de evaluación.

### Opción B: Ejecución Local (Pipeline Modular en Python)

1. **Clonar el repositorio y preparar el entorno virtual:**
   ```bash
   git clone https://github.com/laurat2025/Workshop-3.git
   cd Workshop-3
   python -m venv .venv
   # En Linux/Mac:
   source .venv/bin/activate
   # En Windows:
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configurar variables de entorno:**
   Crea un archivo `.env` en la raíz del proyecto a partir de `.env.example` y completa tus credenciales:
   ```env
   KAFKA_BOOTSTRAP_SERVERS=kafka-classroom-2026.westus.cloudapp.azure.com:9092
   KAFKA_TOPIC=etl.records
   KAFKA_GROUP_ID=laura-vanessa-torres-valencia
   KAFKA_USERNAME=classroom-students
   KAFKA_PASSWORD=TU_PASSWORD_AQUI
   ```

3. **Ejecutar el pipeline ETL + ML completo:**
   ```bash
   python main.py
   ```
   *(Este comando creará automáticamente las carpetas locales `data/` y `models/`, almacenará los resultados en SQLite y guardará el modelo entrenado).*

4. **Auditoría SQL y Dashboard Interactivo:**
   * Ejecuta las consultas de verificación contenidas en `sql/queries.sql` sobre `data/happiness_predictions.db`.
   * Inicia el panel de monitoreo en tiempo real con Streamlit:
     ```bash
     streamlit run src/dashboard.py
     ```