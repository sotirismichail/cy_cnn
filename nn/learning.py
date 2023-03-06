import numpy as np


def generate(out_channels: int = 3, kernel_size: int = 3):
    """
    Description:
        Creates 'number' of kernels, of size kernel_size x kernel_size x kernel_size
        from normally distributed random numbers. Each kernel is normalized to have
        values between 0 and 1, as needed for convolution.

    Parameters:
        out_channels: int (default: 5), output channels of the convolution
        kernel_size: int (default: 3), output channels of the convolution

    Returns:
        kernel_list: Convolution kernels
    """
    kernel_list = []

    for k in range(out_channels):
        kernel = np.around(
            np.random.normal(0, 1, size=(kernel_size, kernel_size, kernel_size)),
            decimals=4,
        )
        kernel = kernel / np.linalg.norm(kernel)
        kernel_list.append(kernel)

    return kernel_list


def bias(
    in_channels: int = 3, out_channels: int = 5, kernel_size: int = 3, groups: int = 1
) -> np.ndarray:
    """
    Description:
        The learnable bias of the module of shape (out_channels). The values of these
        weights are sampled from U(-sqrt(k), sqrt(k)), where
        k = groups/in_channels*kernel_size^3

    Parameters:
        in_channels: int (default: 3), input channels of the convolution
        out_channels: int (default: 3), output channels of the convolution
        kernel_size: int (default: 3), dimensions of convolution kernel
        groups: int (default: 1), convolution channels to be disconnected from the input
                                    to the output

    Returns:
        bias_array: np.ndarray, biases calculated for each kernel
    """

    # bias_array = np.zeros(out_channels)
    kappa = groups / (in_channels * (kernel_size**kernel_size))
    bias_array = np.around(
        np.random.uniform(-np.sqrt(kappa), np.sqrt(kappa), size=out_channels),
        decimals=4,
    )

    return bias_array


def weights(
    in_channels: int = 3, out_channels: int = 5, kernel_size: int = 3, groups: int = 1
) -> np.ndarray:
    """
    Description:
        Calculating the learnable weights of the module of shape (out_channels,
        in_channels/groups, kernel_size, kernel_size). The values of these weights are
        sampled from U(-sqrt(k), sqrt(k)), where k = groups/in_channels*kernel_size^3

    Parameters:
        in_channels: int (default: 3), input channels of the convolution
        out_channels: int (default: 3), output channels of the convolution
        kernel_size: int (default: 3), dimensions of convolution kernel
        groups: int (default: 1), convolution channels to be disconnected from the input
                                    to the output

    Returns:
        weight_array: np.ndarray, weights calculated for each kernel
    """
    # weight_array = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
    kappa = groups / (in_channels * (kernel_size**kernel_size))
    weight_array = np.around(
        np.random.uniform(
            -np.sqrt(kappa),
            np.sqrt(kappa),
            size=(out_channels, in_channels, kernel_size, kernel_size),
        ),
        decimals=4,
    )

    return weight_array
