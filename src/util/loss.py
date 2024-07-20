import torch
import torch.nn as nn


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class WeightedMSE(nn.Module):
    def __init__(self):
        super(WeightedMSE, self).__init__()

        self.sample_prios = torch.empty(0)

    def forward(self, pred, true, weights=None):
        if weights is None:
            return nn.functional.mse_loss(pred, true)
        
        weights = torch.from_numpy(weights).to(DEVICE)
        losses = weights * (pred - true) ** 2
        self.sample_prios = (losses + 1e-5).data.cpu().numpy()
        return losses.mean()