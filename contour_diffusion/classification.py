import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, input_size):
        super(ClassificationHead, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 256),
                                    nn.Linear(256, 32))
        self.proj = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.proj(x)
        x = self.sigmoid(x)
        return x
