from typing import Any, Union

import numpy as np
import config
from filtes.bilinear import interp_bilinear


def _lpcoords(ishape: np.ndarray,
              angles_eval: np.ndarray,
              angles: int = None) -> tuple[Any, Any, Union[int, Any], Any]:
    """
    Description:
        Calculate the reverse coordinates for the log-polar transform.

    Parameters:
        ishape: array, shape of the image processed
        angles_eval: ndarray (ftype), angles of evaluation
        angles: int, number of samples in the radial direction

    Returns:
        An array of shape (len(angles), angles_eval), containing the coordinates, as well as the angles and the
        logarithmic base used to calculate them.
    """
    ishape = np.array(ishape)
    centre = (ishape[:2]-1)/2.

    d = np.hypot(*(ishape[:2]-centre))
    log_base = np.log(d)/angles_eval

    if angles is None:
        angles = -np.linspace(0, 2*np.pi, 2*angles_eval+1)[:-1]
    theta = np.empty((len(angles), angles_eval), config.ftype)
    theta.transpose()[:] = angles
    log_e = np.empty_like(theta)
    log_e[:] = np.arange(angles_eval).astype(config.ftype)

    r = np.exp(log_e*log_base)

    return r*np.sin(theta) + centre[0], r*np.cos(theta) + centre[1], angles, log_base

  
def logpolar(image: np.ndarray,
             output: np.ndarray,
             angles: int = None,
             angles_eval: np.ndarray = None,
             mode: str ='M',
             cval: int = 0,
             _coords_r = None,
             _coords_c = None,
             verbose: bool = False) -> Union[tuple[Any, Union[int, Any], Any], Any]:
    """
    Description:
        Transforms an image from a cartesian (x, y) representation, to a polar representation (r, φ, θ),
        on a logarithmic base.

    Parameters:
        image: ndarray, HxWxC three-dimensional RGB image
        angles: int, number of samples in the radial direction
        angles_eval: ndarray (float), angles of evaluation
        mode: string, how values outside the borders are handled, 'C' for constant, 'M' for mirror and 'W' for wrap
        cval: int/float, constant to fill the outside area with if mode=='C'
        verbose: Set to 'True' to also return the angles and log base to the caller

    Returns:
         lpt: ndarray (uint8), log polar transform of the input image
         angles: ndarray (float), angles used, only if extra_info is set to 'True'
         log_base: int, log base of the transform, only returned if verbose is 'True'
    """
    if angles_eval is None:
        angles_eval = max(image.shape[:2])

    if _coords_r is None or _coords_c is None:
        _coords_r, _coords_c, angles, log_base = _lpcoords(image.shape, angles_eval, angles)

    channels = image.shape[2]
    if output is None:
        output = np.empty(_coords_r.shape + (channels,), dtype=np.uint8)
    else:
        output = np.atleast_3d(np.ascontiguousarray(output))

    for channel in range(channels):
        output[..., channel] = interp_bilinear(image[..., channels], _coords_r, _coords_c, mode=mode, cval=cval,
                                               output=output[..., channel])

    output = output.squeeze()

    if verbose:
        return output, angles, log_base
    else:
        return output
