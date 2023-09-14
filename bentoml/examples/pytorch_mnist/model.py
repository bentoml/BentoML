import torch
import torch.nn as nn


class SimpleConvNet(nn.Module):
    """
    Simple Convolutional Neural Network
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(26 * 26 * 10, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

    def forward(self, x):
        return self.layers(x)

    def predict(self, inp):
        """predict digit for input"""
        self.eval()
        with torch.no_grad():
            raw_output = self(inp)
            _, pred = torch.max(raw_output, 1)
            return pred
