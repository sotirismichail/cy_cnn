import numpy as _np
"""
    config.py

    Configuration file, defining program-running utilities and data types that are used throughout the program. Any
    changes to data types or formats should be made here and not hard-coded in other project files.
"""
__all__ = ['ftype', 'itype', 'utype', 'get_log', 'leaky_relu_slope_coefficient']
"""
    Type definitions
"""
ftype = _np.float64
itype = _np.int32
utype = _np.uint8
"""
    Coefficients
"""
leaky_relu_slope_coefficient = 0.1


def get_log(name: str = "debug_log"):
    """
    Description:
        Instantiate a debugging logger

    Parameters:
        name: str, name of the logger to be created

    Returns:
        logging.getLogger(): object, a logger instance
    """
    import logging
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    return logging.getLogger(name)