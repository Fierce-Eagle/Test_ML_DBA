import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 1, 1, 1, 1),  # [1, 128, 128]
            nn.Conv2d(1, 8, 1, 1, 1),  # [8, 128, 128]
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [8, 64, 64]
            nn.ReLU(),

            nn.Conv2d(8, 16, 3, 1, 0),  # [16, 62, 62]
            nn.Conv2d(16, 32, 3, 1, 0),  # [32, 60, 60]
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 30, 30]
            nn.ReLU(),

            nn.Conv2d(32, 16, 3, 1, 0),  # [16, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, 1, 0),  # [8, 12, 12]
            nn.ReLU(),
            nn.Dropout()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 12 * 12, 144),
            nn.ReLU(),
            nn.Linear(144, 7)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x
