import os
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from utils.device import get_device
from datasets.jena import get_jena_loaders
from models.lstm import LSTMForecaster
from models.rnn import RNNForecaster
from engine.trainer import fit
from callbacks.early_stopping import EarlyStopping

def train_and_log(model, model_name, train_loader, val_loader, criterion, device, epochs, lr, wd):
    """Entrena un modelo específico y registra sus métricas en MLflow."""
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({"model_type": model_name, "learning_rate": lr, "batch_size": train_loader.batch_size})

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        early_stopping = EarlyStopping(patience=6, min_delta=1e-3)

        # Ejecuta el entrenamiento
        history = fit(model, train_loader, val_loader, criterion, optimizer, device, epochs, early_stopping, scheduler)

        # Extraer la mejor pérdida de validación registrada por el Early Stopping
        best_val_loss = early_stopping.best_loss
        mlflow.log_metric("best_val_loss", best_val_loss)

        # Opcional: Registrar el modelo en el servidor de MLflow
        # Mute the Git warnings and fix the model serialization format
        os.environ["GIT_PYTHON_REFRESH"] = "quiet"

        mlflow.pytorch.log_model(
           model,
            artifact_path=f"model_{model_name.lower()}",
            serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE
        )

        return best_val_loss, model

def main() -> None:
    batch_size = 8
    learning_rate = 1e-3
    epochs = 100
    weight_decay = 1e-4

    device = get_device()
    print(f"Training on {device}")

    # Configuración del servidor de MLflow (apunta al contenedor de Docker Compose)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("P2_Forecasting_Jena")

    (train_loader, val_loader, test_loader, mean, std) = get_jena_loaders(
        data_dir="data", batch_size=batch_size, sequence_length=24
    )

    criterion = nn.MSELoss()

    print("--- Entrenando RNN ---")
    rnn_model = RNNForecaster(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(device)
    rnn_loss, trained_rnn = train_and_log(
        rnn_model, "RNN", train_loader, val_loader, criterion, device, epochs, learning_rate, weight_decay
    )

    print("--- Entrenando LSTM ---")
    lstm_model = LSTMForecaster(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(device)
    lstm_loss, trained_lstm = train_and_log(
        lstm_model, "LSTM", train_loader, val_loader, criterion, device, epochs, learning_rate, weight_decay
    )

    # Selección automática del Mejor Modelo
    print(f"\nResultados -> RNN Loss: {rnn_loss:.4f} | LSTM Loss: {lstm_loss:.4f}")
    if lstm_loss < rnn_loss:
        print("Mejor modelo seleccionado: LSTM")
        best_model = trained_lstm
    else:
        print("Mejor modelo seleccionado: RNN")
        best_model = trained_rnn

    # Exportar el artefacto para la etapa de 'Docker Inferencia'
    os.makedirs("artifacts", exist_ok=True)
    torch.save(best_model.state_dict(), "artifacts/best_model.pth")
    print("Exportación exitosa en artifacts/best_model.pth")

if __name__ == "__main__":
    main()
