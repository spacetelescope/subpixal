"""
This module provides utility functions for use by :py:mod:`subpixal` module.

:Author: Mihai Cara (for help, contact `HST Help Desk <https://hsthelp.stsci.edu>`_)

:License: :doc:`../LICENSE`

"""
from __future__ import (absolute_import, division, unicode_literals,
                        print_function)

import numpy as np

from . import __version__, __version_date__


__all__ = ['parse_file_name', 'py2round']


def parse_file_name(image_name):
    """
    Parse image file names including possible extensions.

    Parameters
    ----------
    image_name : str
        An image file name and (optionally) extension specification,
        e.g.: ``'j1234567q_flt.fits[1]'``, ``'j1234568q_flt.fits[sci,2]'``,
        etc.

    Returns
    -------
    file_name : str
        File name itself **without** extension specification.

    ext : tuple, int, None
        A tuple of two elements: *extension name* (a string) and *extension
        version* (an integer number), e.g., ``('SCI', 2)``. Alternatively,
        an extention can be  specified using an integer *extension number*.
        When no extension was specified, ``ext`` returns `None`.

    Examples
    --------
>>> import subpixal
>>> subpixal.parse_file_name('j1234568q_flt.fits[sci,2]')
('SCI', 2)

    """
    numbra = image_name.count('[')
    numket = image_name.count(']')

    if numbra + numket > 0:
        image_name = image_name.rstrip()

    if ((numbra == 1 and ']' != image_name[-1]) or numbra != numket or
        numbra > 1):
        raise ValueError("Misplaced, unbalanced, or nested "
                         "brackets have been detected.")

    if numbra == 0:
        return image_name, None

    idx_bra = image_name.find('[')

    if idx_bra == 0:
        raise ValueError("No valid file name provided.")

    # separate file name from FITS extension specification:
    file_name = image_name[:idx_bra]
    extcomp = image_name[idx_bra + 1:-1].split(',')

    if len(extcomp) == 1:
        # extension specification is an integer extension number:
        try:
            extnum = int(extcomp[0])
        except ValueError:
            raise ValueError("Invalid extension specification.")
        return file_name, extnum

    elif len(extcomp) == 2:
        # extension specification is a tuple of extension name and version:
        try:
            extnum = int(extcomp[1])
        except ValueError:
            raise ValueError("Invalid extension specification.")

        extname = extcomp[0].strip()

        return file_name, (extname, extnum)

    else:
        raise ValueError("Invalid extension specification.")


def py2round(x):
    """
    This function returns a rounded up value of the argument, similar
    to Python 2.
    """
    if hasattr(x, '__iter__'):
        rx = np.empty_like(x)
        m = x >= 0.0
        rx[m] = np.floor(x[m] + 0.5)
        m = np.logical_not(m)
        rx[m] = np.ceil(x[m] - 0.5)
        return rx

    else:
        if x >= 0.0:
            return np.floor(x + 0.5)
        else:
            return np.ceil(x - 0.5)
