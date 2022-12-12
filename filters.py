import numpy as np
import config


def coord_map(dim: int,
              coord: config.ftype,
              mode: str):
    """
    Description:
        Handles the mirror and warp modes of the image interpolation.

    Parameters:
        dim: dimension (row or column) to process
        coord: coordinate of the matrix to calculate
        mode: 'W' for 'warp' or 'm' for mirror bilinear interpolation mode

    Returns:
        coord: ftype (float64, see config), coordinate value, depending on the mode of the bilinear interpolation
    """
    # 12/12/2022 todo: coord seems to be a matrix
    if mode == 'M':
        if coord < 0:
            coord = np.fmod(-coord, dim)
        elif coord == dim:
            coord = dim - 1
        else:
            coord = dim - np.fmod(coord, dim)
    elif mode == 'W':
        if coord < 0:
            coord = dim - np.fmod(-coord, dim)
        elif coord == dim:
            coord = 0
        else:
            coord = np.fmod(coord, dim)

    return coord


def interp_bilinear(img_channel: np.ndarray,
                    tf_coords_r: np.ndarray = None,
                    tf_coords_c: np.ndarray = None,
                    mode: str = 'N',
                    cval: int = 0) -> np.ndarray:
    """
    Description:
        Bilinear interpolation filter, with three modes of interpolation:
        'C', constant, with the constant being 'cval', set by the input, 'W', warp, and 'M', mirror

    Parameters:
        img_channel: One channel of the input image
        output: the result of the interpolation
        tf_coords_r: row transform coordinates
        tf_coords_c: column transform coordinates
        mode: interpolation mode, 'C' sets the values equal to 'cval', 'W' warps and 'M' mirrors
        cval: the value for the constant value interpolation mode

    Returns:
        output: np.ndarray, a channel with applied interpolation
    """

    img_channel = img_channel.astype(config.utype)
    tf_coords_r = tf_coords_r.astype(config.ftype)
    tf_coords_c = tf_coords_c.astype(config.ftype)

    output = np.empty(tf_coords_r.shape, dtype=config.utype)

    rows, columns = img_channel.shape
    tf_rows, tf_columns = tf_coords_r.shape

    for tfr in range(tf_rows):
        for tfc in range(tf_columns):
            r = tf_coords_r[tfr*tf_columns + tfc]
            c = tf_coords_c[tfr*tf_columns + tfc]

            if ((mode == 'C') and ((r < 0) or (r >= rows) or
                                   (c < 0) or (c >= columns))):
                output[tfr * tf_columns + tfc] = cval
            else:
                r = coord_map(rows, r, mode)
                c = coord_map(columns, c, mode)

                r_int = np.floor(r)
                c_int = np.floor(c)

                t = r - r_int
                u = c - c_int

                y0 = img_channel[int(r_int * columns + c_int)]
                y1 = img_channel[int(coord_map(rows, r_int + 1, mode) * columns + c_int)]
                y2 = img_channel[int(coord_map(rows, r_int + 1, mode) * columns + coord_map(columns, c_int + 1, mode))]
                y3 = img_channel[int(r_int * columns + coord_map(columns, c_int + 1, mode))]

                output[tfr * tf_columns + tfc] = \
                    (1 - t) * (1 - u) * y0 + t * (1 - u) * y1 + t * u * y2 + (1 - t) * u * y3

    return output
