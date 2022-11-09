import numpy as np
from typing import List, Tuple, Union


def cy_padding(matrix: np.ndarray,
               padding: Tuple[int, int]) -> np.ndarray:
    n, m = matrix.shape
    r, c = padding

    matrix_padded = np.zeros((n + r * 2, m + c * 2))
    matrix_padded[r:n + r, c:m + c] = matrix
    for i in range(r):
        matrix_padded[0:r, c:m + c] = np.flipud(matrix[n - r:n, :])
        matrix_padded[n + r:n + 2 * r, c:m + c] = matrix[0:r, :]

    return matrix_padded


def cy_conv2d(matrix: Union[List[List[float]], np.ndarray],
              kernel: Union[List[List[float]], np.ndarray],
              stride: Tuple[int, int] = (1, 1),
              dilation: Tuple[int, int] = (1, 1),
              padding: Tuple[int, int] = (0, 0)) -> np.ndarray:
    k = kernel.shape
    n, m = matrix.shape

    matrix = matrix if list(padding) == [0, 0] else cy_padding(matrix, padding)

    h_out = np.floor((n + 2 * padding[0] - k[0] - (k[0] - 1) * (dilation[0] - 1)) / stride[0]).astype(int) + 1
    w_out = np.floor((m + 2 * padding[1] - k[1] - (k[1] - 1) * (dilation[1] - 1)) / stride[1]).astype(int) + 1
    matrix_conv = np.zeros((h_out, w_out))

    b = k[0] // 2, k[1] // 2
    centre_x_0 = b[0] * dilation[0]
    centre_y_0 = b[1] * dilation[1]
    for i in range(h_out):
        centre_x = centre_x_0 + i * stride[0]
        indices_x = [centre_x + l * dilation[0] for l in range(-b[0], b[0] + 1)]
        for j in range(w_out):
            centre_y = centre_y_0 + j * stride[1]
            indices_y = [centre_y + l * dilation[1] for l in range(-b[1], b[1] + 1)]

            submatrix = matrix[indices_x, :][:, indices_y]

            matrix_conv[i][j] = np.sum(np.multiply(submatrix, kernel))
    return matrix_conv


def apply_conv(image: np.ndarray,
               kernel: List[List[float]]) -> np.ndarray:
    kernel = np.asarray(kernel)
    b = kernel.shape
    return np.dstack([cy_conv2d(image[:, :, z], kernel, padding=(b[0] // 2, b[1] // 2))
                      for z in range(3)]).astype('uint8')
