Sistema de pronóstico de temperatura basado en redes neuronales recurrentes (RNN y LSTM) implementado bajo una arquitectura MLOps reproducible y contenedorizada.

## Arquitectura y Stack Tecnológico
* **Deep Learning:** PyTorch (entrenamiento competitivo entre arquitecturas `RNNForecaster` y `LSTMForecaster` con selección automática del mejor modelo mediante pérdida de validación).
* **Experiment Tracking:** MLflow (registro de métricas, hiperparámetros y serialización de artefactos).
* **Despliegue e Inferencia:** API REST desarrollada con FastAPI y servida con Uvicorn en un contenedor aislado (`Dockerfile.inferencia`).
* **Orquestación:** Docker Compose con aislamiento de red (*Sidecar Network*) entre entrenamiento y servidor de tracking.
* **Testing y Calidad:** Suite de pruebas unitarias y de integración con `pytest` y validación de formato automatizada mediante `pre-commit`.
* **Versionado de Datos:** DVC (`Dataset_V1`).

## Ejecución Rápida

### 1. Entrenamiento y Tracking
Levantar el servidor de MLflow y ejecutar el pipeline de entrenamiento competitivo:
```bash
docker compose up -d mlflow
docker compose run train
