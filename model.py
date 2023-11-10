import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 1, 1, 1, 1),  # [1, 512, 512]
            nn.Conv2d(1, 64, 1, 1, 1),  # [64, 512, 512]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 0),  # [64, 510, 510]
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 260, 260]
            nn.ReLU(),
            nn.Dropout(),

            nn.Conv2d(64, 32, 3, 1, 0),  # [32, 208, 208]
            nn.Conv2d(32, 32, 3, 1, 0),  # [32, 206, 206]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 0),  # [32, 204, 204]
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [32, 102, 102]
            nn.ReLU(),
            nn.Dropout(),

            nn.Conv2d(32, 16, 3, 1, 0),  # [16, 100, 100]
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [16, 50, 50]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 0),  # [16, 48, 48]
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [16, 24, 24]
            nn.ReLU(),
            nn.Dropout()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 12 * 12, 1024),  # [2304, 1024]
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x
