from torch import nn
from tpsmm_plus.modules.util import Hourglass


class DepthGenerator(nn.Module):
    def __init__(self):
        super(DepthGenerator, self).__init__()

        self.hourglass = Hourglass(
            block_expansion=64, in_features=3, max_features=1024, num_blocks=5
        )
        self.to_depth = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        depth = self.hourglass(x)
        