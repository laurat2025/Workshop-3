-- Consultas de verificación sobre la base de datos SQLite generada

-- 1. Resumen de errores de predicción en el set de evaluación
SELECT 
    COUNT(*) AS total_evaluados,
    ROUND(AVG(actual_score), 4) AS promedio_real,
    ROUND(AVG(predicted_score), 4) AS promedio_predicho,
    ROUND(AVG(abs_error), 4) AS mae_global
FROM happiness_predictions;

-- 2. Top 10 registros con mayor error absoluto
SELECT *
FROM happiness_predictions
ORDER BY abs_error DESC
LIMIT 10;

-- 3. Monitoreo de inferencia en streaming (si se ejecutó src/streaming_inference.py)
SELECT 
    country,
    year,
    actual_score,
    predicted_score,
    abs_error,
    processed_at
FROM streaming_inference_log
ORDER BY id DESC
LIMIT 20;
