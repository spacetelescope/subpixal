# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides utility functions for use by :py:mod:`subpixal` module.

:Author: Mihai Cara (for help, contact `HST Help Desk <https://hsthelp.stsci.edu>`_)

:License: :doc:`LICENSE`

"""
import tempfile

import numpy as np
from astropy.io import fits

from . import __version__, __version_date__


__all__ = ['parse_file_name', 'py2round', 'get_ext_list']


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
    >>> subpixal.utils.parse_file_name('j1234568q_flt.fits[sci,2]')
    ('j1234568q_flt.fits', ('sci', 2))

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


def get_ext_list(image, extname='SCI'):
    """
    Return a list of all extension versions of `extname` extensions.
    `image` can be either a file name or a `astropy.io.fits.HDUList` object.

    This function returns a list of fully qualified extensions: a list of
    tuples of the form (``'extname'``, ``'extver'``).

    Examples
    --------
    >>> get_ext_list('j9irw1rqq_flt.fits')
    [('SCI', 1), ('SCI', 2)]

    """
    if not isinstance(extname, str):
        raise TypeError("Argument 'extname' must be either a string "
                        "indicating the value of the 'EXTNAME' keyword of the "
                        "extensions whose versions are to be returned.")

    extname = extname.upper()

    close = False

    try:
        if isinstance(img, (str, bytes)):
            image = fits.open(image, mode='update')
            close = True

        elif not isinstance(image, fits.HDUList):
            raise TypeError("Argument 'imgage' must be either a file name, "
                            "or an astropy.io.fits.HDUList object.")

        ext = []
        for e in hdulist:
            hdr = e.header
            if 'EXTNAME' in hdr and hdr['EXTNAME'].upper() == extname:
                ext.append((extname, hdr['EXTVER'] if 'EXTVER' in hdr else 1))

    except:
        raise

    finally:
        if close:
            image.close()

    return ext


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


def _create_tmp_fits_file(image, prefix='tmp_'):
    tmpf = None
    close_image = False
    try:
        if isinstance(image, np.ndarray):
            image = fits.PrimaryHDU(image)

        elif not isinstance(image, fits.HDUList):
            image = fits.open(image)
            close_image = True

        tmpf = tempfile.NamedTemporaryFile(
            mode='wb', suffix='.fits', prefix=prefix, dir='./',
            delete=True,
        )

        image.writeto(tmpf)
        tmpf.file.flush()
        tmpf.file.seek(0)

    except:
        if tmpf is not None:
            tmpf.close()
        raise

    finally:
        if close_image:
            image.close()

    return tmpf
