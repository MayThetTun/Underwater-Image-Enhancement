import torch
import torch.nn as nn


# Initialize the SIMAM (Scale-Invariant Multi-Attention Module).
#
# Args:
#     channels (int): Number of input channels. Defaults to None.
#     e_lambda (float): Epsilon value for numerical stability. Defaults to 1e-4.

class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    """
        Forward pass of the SIMAM 
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        Returns:
            torch.Tensor: Output tensor after applying SIMAM
    """

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=58, kernel_size=3, stride=1, padding=1, bias=False)
        self.ca = simam_module() # SIMAM Attention module
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=.20)

    def forward(self, x):
        x, input_x = x
        a = self.ca(self.relu(self.conv1(self.ca(self.relu(self.drop(self.conv(self.relu(self.drop(self.conv(x))))))))))
        out = torch.cat((a, input_x), 1)
        return out, input_x


class UWnet(nn.Module):
    def __init__(self, num_layers=3):
        super(UWnet, self).__init__()
        self.input = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = self.StackBlock(ConvBlock, num_layers)

    def StackBlock(self, block, layer_num):
        layers = []
        for _ in range(layer_num):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        input_x = x
        x1 = self.relu(self.input(x))
        out, _ = self.blocks((x1, input_x))
        out = self.output(out)
        return out
