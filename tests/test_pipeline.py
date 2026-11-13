import torch
from fastapi.testclient import TestClient
from models.rnn import RNNForecaster
from models.lstm import LSTMForecaster
from app import app

client = TestClient(app)

# Prueba 1: Validar dimensionalidad y salida de RNNForecaster
def test_rnn_forward_shape():
    model = RNNForecaster(input_size=1, hidden_size=50, num_layers=2, output_size=1)
    dummy_input = torch.randn(4, 24, 1)  # [batch_size=4, seq_len=24, input_size=1]
    output = model(dummy_input)

    assert output.shape == (4, 1), f"Forma esperada (4, 1), obtenida {output.shape}"
    assert not torch.isnan(output).any(), "La salida de RNN contiene valores NaN"

# Prueba 2: Validar dimensionalidad y salida de LSTMForecaster
def test_lstm_forward_shape():
    model = LSTMForecaster(input_size=1, hidden_size=50, num_layers=2, output_size=1)
    dummy_input = torch.randn(4, 24, 1)  # [batch_size=4, seq_len=24, input_size=1]
    output = model(dummy_input)

    assert output.shape == (4, 1), f"Forma esperada (4, 1), obtenida {output.shape}"
    assert not torch.isnan(output).any(), "La salida de LSTM contiene valores NaN"

# Prueba 3: Validar código HTTP 200 y respuesta JSON en la API de inferencia
def test_api_predict_endpoint():
    payload = {
        "sequence": [
            10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.2, 11.5,
            11.8, 12.0, 12.2, 12.5, 12.8, 13.0, 13.1, 13.2,
            13.3, 13.2, 13.0, 12.8, 12.5, 12.2, 11.9, 11.5
        ]
    }
    response = client.post("/predict", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert "temperatura_predicha" in data
    assert isinstance(data["temperatura_predicha"], float)
