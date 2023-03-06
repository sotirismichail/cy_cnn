import numpy as np
import conv

from nn.learning import weights


# TODO: implement forward and backward propagation for the neural network
class cyConvFunction():
    @staticmethod
    def forward(input, weight, workspace, stride=1, padding=0, dilation=1):
        pass
    
    @staticmethod
    def backward(grad_output):
        pass


# TODO: write tests for the classes
# TODO: write documentation for the layer class
class cyConv():
    workspace = np.zeros(1024 * 1024 * 1024 * 1, dtype=np.float32)

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
    ) -> None:
        super(cyConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = weights(in_channels, out_channels, kernel_size)

    def forward(self, input):
        output = cyConvFunction.apply(input, self.weight, cyConv.workspace, self.stride, self.padding, self.dilation)
        return output