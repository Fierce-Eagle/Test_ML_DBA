import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 1, 1, 1, 1),  # [1, 256, 256]
            nn.Conv2d(1, 16, 1, 1, 1),  # [16, 256, 256]
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [16, 128, 128]
            nn.ReLU(),
            nn.Dropout(),

            nn.Conv2d(16, 32, 3, 1, 0),  # [16, 126, 126]
            nn.Conv2d(32, 32, 3, 1, 0),  # [32, 124, 124]
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [32, 62, 62]
            nn.ReLU(),
            nn.Dropout(),

            nn.Conv2d(32, 16, 3, 1, 0),  # [16, 60, 60]
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [16, 30, 30]
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, 1, 0),  # [8, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [8, 14, 14]
            nn.ReLU(),
            nn.Dropout()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 14 * 14, 144),
            nn.ReLU(),
            nn.Linear(144, 7),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x
