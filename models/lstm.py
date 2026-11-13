import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        num_layers=1,
        output_size=1,
        ):
        super(LSTMForecaster, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.output = nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
        )

    def forward(self, x):
        output, (hidden, cell) = self.rnn(x)

        #[batch_size, seq_len, hidden_size]
        last_hidden_state = hidden[-1]
        return self.output(last_hidden_state)
