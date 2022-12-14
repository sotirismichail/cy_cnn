import numpy as np
import config


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Description:
        The standard bounded logistical function, a sigmoid curve, used in neural networks as an activation function.

    Parameters:
        z: np.ndarray, 2D data, output of the affine transformation

    Returns:
        activation: np.ndarray, post-activation output
    """
    return 1/(1 + np.exp(-z))


def tanh(z: np.ndarray) -> np.ndarray:
    """
    Description:
        Hyperbolic tangent activation function, offers a mean closer to zero compared to the standard bounded logistical
        function, which means the centre of the output of the activation points is also closer to zero, and therefore
        the range of the values is smaller, and the learning time decreases.

    Parameters:
        z: np.ndarray, 2D data, output of the affine transformation

    Returns:
        activation: np.ndarray, post-activation output
    """
    return np.tanh(z)


def relu(z) -> np.ndarray:
    """
    Description:
        Rectified linear activation unit function, a linear function when the values are greater than zero and
        non-linear for values smaller than zero, since the function always returns 0 for negative values.

    Parameters:
        z: np.ndarray, 2D data, output of the affine transformation

    Returns:
        activation: np.ndarray, post-activation output
    """
    return np.maximum(0, z)


def relu_leaky(z) -> np.ndarray:
    """
    Description:
        Leaky rectified linear unit function, based on ReLU, but provides a small slope for negative values instead of
        a flat slope, to overcome the zero-gradient issue of the standard ReLU. The slope coefficient is pre-determined.
        It can be adjusted through the config.py file.

    Parameters:
        z: np.ndarray, 2D data, output of the affine transformation

    Returns:
        activation: np.ndarray, post-activation output
    """
    return np.maximum(config.leaky_relu_slope_coefficient * z, z)

