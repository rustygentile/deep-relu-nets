import torch.nn as nn
from torch.nn import functional


class ReLUNet(nn.Module):
    """
    ReLU neural network with a fixed width and depth
    """
    def __init__(self, width, depth):

        super().__init__()
        self.width = width
        self.depth = depth

        self.layer_first = nn.Linear(1, width)
        self.layers = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(depth)]
        )
        self.layer_last = nn.Linear(width, 1)

    def part(self, x, max_layer):

        x = functional.relu(self.layer_first(x))

        for l in self.layers[:max_layer]:
            x = functional.relu(l(x))

        return x

    def forward(self, x):

        max_layer = len(self.layers)
        x = self.part(x, max_layer)
        x = self.layer_last(x)

        return x
