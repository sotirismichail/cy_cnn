import numpy as _np
"""
    config.py
    
    Configuration file, defining program-running utilities and data types that are used throughout the program. Any
    changes to data types or formats should be made here and not hard-coded in other project files.
"""
__all__ = ['ftype', 'itype', 'utype', 'get_log']

ftype = _np.float64
itype = _np.int32
utype = _np.uint8


def get_log(name):
    import logging
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    return logging.getLogger(name)
