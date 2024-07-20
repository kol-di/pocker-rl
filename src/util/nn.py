import torch
import torch.nn as nn


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PockerNN(nn.Module):
    def __init__(self, input_dim=5):
        super(PockerNN, self).__init__()
        self.input_dim = input_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self._initialise_weights()

    def forward(self, x):
        x = self._normalise_input(x)
        out = self.fc(x)

        return out

    def _normalise_input(self, x):
        # Predefined ranges
        min_values = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32).to(DEVICE)
        max_values = torch.tensor([50, 50, 1, 12, 1], dtype=torch.float32).to(DEVICE)

        # Min-max
        x_norm = (x - min_values) / (max_values - min_values)

        return x_norm

    def _initialise_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)