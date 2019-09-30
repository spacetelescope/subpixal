# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A module that provides algorithm for creating sub-pixel cross-correlation
images and computing displacements.

:Author: Mihai Cara (for help, contact `HST Help Desk <https://hsthelp.stsci.edu>`_)

:License: :doc:`LICENSE`

"""
import numpy as np
from scipy import signal

from .centroid import find_peak
from . import __version__, __version_date__


__all__ = ['find_displacement']


def find_displacement(ref_image, image00, image10, image01, image11,
                      cc_type='NCC', full_output=False):
    """
    Find subpixel displacements between one reference cutout and a set of
    four "dithered" cutouts. This is achieved by finding peak position in a
    "supersampled" cross-correlation image obtained by interlacing
    cross-correlation maps of reference cutout with each dithered cutout.

    Parameters
    ----------
    ref_image : numpy.ndarray
        Image of a reference cutout.

    image00 : numpy.ndarray
        Image whose displacement relative to reference image needs to be
        computed. It must have same shape as ``ref_image``.

    image10 : numpy.ndarray
        "Same" image as ``image00`` but sampled at a 1/2 pixel displacement
        along the X-axis. It must have same shape as ``ref_image``.

    image01 : numpy.ndarray
        "Same" image as ``image00`` but sampled at a 1/2 pixel displacement
        along the Y-axis. It must have same shape as ``ref_image``.

    image11 : numpy.ndarray
        "Same" image as ``image00`` but sampled at a 1/2 pixel displacement
        along both X-axis and Y-axis. It must have same shape as ``ref_image``.

    cc_type : {'CC', 'NCC', 'ZNCC'}, optional
        Cross-correlation algorithm to be used. ``'CC'`` indicates the
        "standard" cross-correlation algorithm. ``'NCC'`` refers to the
        normalized cross-correlation and ``'ZNCC'`` refers to the
        zero-normalized cross-correlation, see, e.g.,
        `Terminology in image processing \
        <https://en.wikipedia.org/wiki/Cross-correlation#\
        Terminology_in_image_processing>`_.

    full_output : bool, optional
        Return displacements only (``full_output=False``) or also interlaced
        cross-correlation image and direct (non-interlaced) cross-correlation images

    Returns
    -------

    dx : float
        Displacement of ``image00`` with regard to ``ref_image`` along the
        X-axis (columns).

    dy : float
        Displacement of ``image00`` with regard to ``ref_image`` along the
        Y-axis (rows).

    icc : numpy.ndarray, Optional
        Interlaced ("supersampled") cross-correlation image. Returned only when
        ``full_output`` is `True`.

    ccs : list of numpy.ndarray, Optional
        List of cross-correlation images between ``ref_image`` and ``image??``.
        Returned only when ``full_output`` is `True`.

    """
    icc, ccs = _build_icc_image(
        ref_image, image00, image10, image01, image11, cc_type
    )
    xm, ym = find_peak(icc, peak_fit_box=5, peak_search_box='all')

    # find center of the auto-correlation peak when using signal.fftconvolve:
    xc = (icc.shape[1] - 1) // 4
    yc = (icc.shape[0] - 1) // 4

    dx = 0.5 * xm - xc
    dy = 0.5 * ym - yc

    return (dx, dy, icc, ccs) if full_output else (dx, dy)


def _build_icc_image(ref, im00, im10, im01, im11, cc_type='NCC'):
    """ Build interlaced ("oversampled") cross-correlation image. """
    # Check that all images have the same size. This is not really required,
    # but it simplifies things and, moreover, the rest of the code is designed
    # to produce images of the same size:
    if not np.all(np.equal(ref.shape,
                           [im00.shape, im10.shape, im01.shape, im11.shape])):
        raise ValueError("All cutouts must have same shape.")

    cc_type = cc_type.upper()
    if cc_type in ['ZNCC', 'NCC']:
        ref, [im00, im10, im01, im11] = _normalize(
            ref, [im00, im10, im01, im11], cc_type == 'ZNCC'
        )

    # cross-correlate images of different shifts
    cc00 = signal.fftconvolve(ref, im00[::-1, ::-1], mode='same')
    cc10 = signal.fftconvolve(ref, im10[::-1, ::-1], mode='same')
    cc01 = signal.fftconvolve(ref, im01[::-1, ::-1], mode='same')
    cc11 = signal.fftconvolve(ref, im11[::-1, ::-1], mode='same')

    # combine/interlace cross-correlated images into
    # an "over-sampled" cross-correlation image:
    ny, nx = cc00.shape
    icc = np.empty((2 * ny, 2 * nx), dtype=cc00.dtype.type)
    icc[::2, ::2] = cc00[::-1, ::-1]
    icc[::2, 1::2] = cc10[::-1,::-1]
    icc[1::2, ::2] = cc01[::-1, ::-1]
    icc[1::2, 1::2] = cc11[::-1, ::-1]

    return icc, (cc00, cc10, cc01, cc11)


def _normalize(ref, images, zero=False):
    data = []
    masks = []
    for im in images:
        mask = im != 0
        masks.append(mask)
        data.append(im[mask])

    data = np.hstack(data)

    mean = np.mean(data) if zero else 0.0
    std = np.std(data)

    norm_images = [im.copy() for im in images]
    for im, m in zip(norm_images, masks):
        if zero:
            im[m] -= mean
        im[m] /= std

    mask = np.sum(masks, axis=0, dtype=np.bool)
    ref = ref.copy()
    if zero:
        ref -= np.mean(ref[mask])
    ref /= np.std(ref[mask])

    return ref, norm_images
