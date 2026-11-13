from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from models.lstm import LSTMForecaster  # Cambia a RNNForecaster si ese fue tu mejor modelo

app = FastAPI(title="API de Inferencia - Forecasting Jena")

# Esquema de datos esperado: una lista de 24 temperaturas (float)
class ForecastRequest(BaseModel):
    sequence: list[float]

# Cargar el modelo estáticamente al arrancar el servidor
device = torch.device("cpu")
model = LSTMForecaster(input_size=1, hidden_size=50, num_layers=2, output_size=1)
model.load_state_dict(torch.load("artifacts/best_model.pth", map_location=device, weights_only=True))
model.eval()

@app.post("/predict")
def predict(request: ForecastRequest):
    # Validar que la secuencia tenga exactamente 24 horas de datos (como en tu train.py)
    if len(request.sequence) != 24:
        raise HTTPException(status_code=400, detail="La secuencia debe contener exactamente 24 valores numéricos.")

    # Convertir la lista de Python a un tensor con forma [batch_size=1, seq_len=24, input_size=1]
    tensor_data = [[[x] for x in request.sequence]]
    input_tensor = torch.tensor(tensor_data, dtype=torch.float32)

    with torch.inference_mode():
        output = model(input_tensor)
        prediction = output.item()

    return {
        "modelo_utilizado": "LSTM", # Cambiar si usas RNN
        "temperatura_predicha": prediction
    }
