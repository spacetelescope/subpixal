"""
A module that provides blotting algorithm for image cutouts and a default
WCS-based coordinate mapping class.

:Author: Mihai Cara (for help, contact `HST Help Desk <https://hsthelp.stsci.edu>`_)

:License: :doc:`../LICENSE`

"""
import copy
import numpy as np
from astropy import wcs
from drizzlepac import cdriz
from stwcs import distortion


__all__ = ['BlotWCSMap', 'blot_cutout']


class BlotWCSMap(object):
    """
    Coordinate mapping class that performs coordinate transformation
    from the source cutout to the "target" cutout.
    The target cutout simply provides a coordinate system. This class
    implements coordinate transformation in the ``__call__()`` method.

    Parameters
    ----------
    source_cutout : Cutout
        A cutout that defines source coordinate system (input to the
        ``__call__(x, y)`` method).

    target_cutout : Cutout
        A cutout that provides target coordinates system to which
        source coordinates need to be mapped.

    """
    def __init__(self, source_cutout, target_cutout):
        self._source_cutout = source_cutout
        self._target_cutout = target_cutout

    def __call__(self, x, y):
        """
        Evaluates transformation from the source coordinates to the
        "target" cutout coordinates using the provided input coordinates
        ``x`` and ``y``.

        .. note::
            Coordinates must be one-based indexed, that is, top-left pixel has
            coordinates ``(1, 1)`` instead of the zero-based indexing used in
            ``numpy`` or ``C``.

        Parameters
        ----------
        x : `numpy.ndarray`, `list`
            A 1D array or list of X-coordinates of points in the image
            coordinate system of the ``from_cutout``.

        y : `numpy.ndarray`, `list`
            A 1D array or list of Y-coordinates of points in the image
            coordinate system of the ``from_cutout``.

        Returns
        -------
        target_coordinates : tuple of two 1D `numpy.ndarray`
            A pair of 1D `numpy.ndarray` containing coordinates mapped to
            the coordinate system of ``target_cutout``.

        """
        target_coordinates = self._target_cutout.world2pix(
            *self._source_cutout.pix2world(x, y, origin=1),
            origin=1
        )

        return target_coordinates


def blot_cutout(source_cutout, target_cutout, interp='poly5', sinscl=1.0,
                wcsmap=None):
    """
    Performs 'blot' operation to create a single blotted image from a
    single source image. All distortion information is assumed to be included
    in the WCS of the ``source_cutout`` and ``target_cutout``.

    Parameters
    ----------
    source_cutout : Cutout
        Cutout that needs to be "blotted". Provides source image for the "blot"
        operation and a WCS.

    target_cutout : Cutout
        Cutout to which ``source_cutout`` will be "blotted". This cutout
        provides a WCS and an output grid.

    interp : {'nearest', 'linear', 'poly3', 'poly5', 'spline3', 'sinc'}, optional
        Form of interpolation to use when blotting pixels.

    sinscl : float, optional
        Scale for sinc interpolation kernel (in output, blotted pixels)

    wcsmap : callable, optional
        Custom mapping class to use to provide transformation from
        source cutout image coordinates to target cutout image coordinates.

    """
    if len(source_cutout.naxis) != 2 or len(target_cutout.naxis) != 2:
        raise ValueError("Cutouts must be 2D images.")

    outsci = np.zeros(tuple(target_cutout.naxis[::-1]), dtype=np.float32)

    misval = 0.0
    kscale = 1.0

    xmin = 1
    ymin = 1
    xmax, ymax = source_cutout.naxis

    if wcsmap is None:
        wcsmap = BlotWCSMap(target_cutout, source_cutout)

    # this pixel scale ratio computation uses a very bad pscale estimate
    # adopted throughout drizzlepac and other HST software. More accurate
    # estimate is to do:
    #
    # pix_ratio = source_cutout.pscale / target_cutout.pscale
    #
    # but then the results would not match drizzlepac.ablot.blot with
    # high accuracy (this does not imply that matching to drizzlepac is
    # the correct thing to do).
    wcslin = distortion.utils.make_orthogonal_cd(target_cutout.wcs)
    pix_ratio = source_cutout.wcs.pscale / wcslin.pscale

    exptime = source_cutout.exptime
    source_data = np.array(source_cutout.data, dtype=np.float32)
    if source_cutout.data_units == 'counts':
        # convert source_cutout.data from counts to rate:
        source_data /= source_cutout.exptime

    cdriz.tblot(
        source_data, outsci,
        xmin, xmax, ymin, ymax,
        pix_ratio, kscale, 1.0, 1.0,
        'center', interp, target_cutout.exptime,
        misval, sinscl, 1, wcsmap
    )

    if target_cutout.data_units == 'rate':
        # convert output blot data from counts to rate
        outsci /= target_cutout.exptime

    out_cutout = copy.deepcopy(target_cutout)
    out_cutout.data[:, :] = outsci

    return out_cutout
