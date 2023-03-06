import numpy as np
from typing import Tuple
import nn


def pad(
    matrix: np.ndarray, padding: Tuple[int, int], mode: str = "zeros"
) -> np.ndarray:
    """
    Description:
        To achieve cylindrical convolution, instead of padding the matrix with zeroes at
        the top and bottom boundaries of elements of the first row (row 0) are copied to
        the padding boundary at the bottom and the elements of the last row (row 7) are
        copied to the boundary at the top of padded matrix. Left and right padding
        remain the same, padded with zeroes.

    Parameters:
        matrix: np.ndarray, convolution layer matrix
        padding: int (default: 1, 1), output channels of the convolution
        mode: str (default: "normal"), padding mode, normal or cylindrical

    Returns:
        matrix_padded: padded matrix for convolution
    """
    n, m = matrix.shape
    r, c = padding

    matrix_padded = np.zeros((n + r * 2, m + c * 2))
    matrix_padded[r : n + r, c : m + c] = matrix

    if mode == "cylindrical":
        for i in range(r):
            matrix_padded[0:r, c : m + c] = np.flipud(matrix[n - r : n, :])
            matrix_padded[n + r : n + 2 * r, c : m + c] = matrix[0:r, :]

        return matrix_padded
    else:
        return matrix_padded


def cyconv2d(
    matrix: np.ndarray,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    padding_mode: str = "zeros",
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
) -> np.ndarray:
    """
    Description:
        Applies a 2D convolution over the input matrix composed of several input planes.
        A cylindrical padding is applied for the input data, so the convolution is
        implementing a cylindrical sliding window algorithm.

    Parameters:
        matrix: np.ndarray, image data as a numpy matrix, 4 dimensions,
                (1, channels, width, height)
        in_channels: int (default: 3), input channels of the convolution
        out_channels: int (default: 3), output channels of the convolution
        kernel_size: int (default: 3), dimensions of convolution kernel
        stride: int (default: 1), stride of convolution
        padding: int (default: 0), padding added to all four sides of the input
        padding_mode: str (default: zeros), 'zeros', 'cylindrical', 'valid'
        dilation: int (default: 1), spacing between kernel elements
        groups: int (default: 1), connections to be blocked from the input to the output
        bias: bool (default: True), adds a learnable bias to the output

    Returns:
        result: np.ndarray, result of the convolution
    """
    in_height, in_width, channels = matrix.shape
    workspace = np.zeros((in_height + 2 * padding, in_width + 2 * padding, channels))

    for channel in range(channels):
        workspace[:, :, channel] = pad(
            matrix[:, :, channel], (padding, padding), mode=padding_mode
        )

    workspace = np.transpose(workspace, (2, 0, 1))
    workspace = np.expand_dims(workspace, axis=0)
    workspace = workspace / 255.0

    weights = nn.learning.weights(in_channels, out_channels, kernel_size, groups)
    biases = nn.learning.bias(in_channels, out_channels, kernel_size, groups)

    out_width = np.fix(
        ((in_width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    )
    out_height = np.fix(
        ((in_height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    )
    result = np.zeros((1, out_channels, out_width, out_height))

    for out_width_iter in range(out_width):
        for out_height_iter in range(out_height):
            for out_channel_iter in range(out_channels):
                total = 0
                for in_channel_iter in range(in_channels):
                    kernel_w = 0
                    for kernel_width_iter in range(kernel_size):
                        for kernel_height_iter in range(kernel_size):
                            element_weight = weights[
                                out_channel_iter,
                                in_channel_iter,
                                kernel_width_iter,
                                kernel_height_iter,
                            ]
                            conv_w = kernel_width_iter + out_width_iter * stride
                            conv_h = kernel_height_iter + out_height_iter * stride
                            val = workspace[0, in_channel_iter, conv_h, conv_w]
                            kernel_w += element_weight * val
                    total += kernel_w
                if bias:
                    result[0, out_channel_iter, out_width_iter, out_height_iter] = (
                        total + biases[out_channel_iter]
                    )
                else:
                    result[0, out_channel_iter, out_width_iter, out_height_iter] = total

    return result
