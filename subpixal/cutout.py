# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A module that provides tools for creating and mapping image cutouts.

:Author: Mihai Cara (for help, contact `HST Help Desk <https://hsthelp.stsci.edu>`_)

:License: :doc:`LICENSE`

"""
import numpy as np
from astropy.io import fits
from astropy import wcs as fitswcs
from stwcs.wcsutil import HSTWCS
from stsci.tools.bitmask import bitfield_to_boolean_mask

from .catalogs import ImageCatalog


__all__ = ['Cutout', 'create_primary_cutouts', 'create_cutouts',
           'NoOverlapError', 'PartialOverlapError']


def _ceil(v):
    vi = int(v)
    if v > vi:
        vi += 1
    return vi


def _floor(v):
    vi = int(v)
    if v < vi:
        vi -= 1
    return vi


class NoOverlapError(ValueError):
    """ Raised when cutout does not intersect the extraction image. """
    pass


class PartialOverlapError(ValueError):
    """ Raised when cutout only partially overlaps the extraction image. """
    pass


def create_primary_cutouts(catalog, segmentation_image, imdata, imwcs,
                           imdq=None, dqbitmask=0, imweight=None,
                           data_units='counts', exptime=1, pad=1):
    """
    A function for creating first-order cutouts from a (drizzle-)combined
    image given a source catalog and a segmentation image.

    Parameters
    ----------
    catalog : ImageCatalog, astropy.table.Table
        A table of sources which need to be extracted. ``catalog`` must contain
        a column named ``'id'`` which contains IDs of segments from the
        ``segmentation_image``. If ``catalog`` is an `astropy.table.Table`,
        then it's ``meta`` attribute may contain an optional
        ``'weight_colname'`` item indicating which column in the table shows
        source weight. If not provided, unweighted fitting will be performed.

    segmentation_image: numpy.ndarray
        A 2D segmentation image identifying sources from the catalog
        in ``imdata``.

    imdata: numpy.ndarray
        Image data array.

    imwcs: astropy.wcs.WCS
        World coordinate system of image ``imdata``.

    imdq: numpy.ndarray, None, optional
        Data quality (DQ) array corresponding to ``imdata``.

    dqbitmask : int, str, None, optional
        Integer sum of all the DQ bit values from the input ``imdq``
        DQ array that should be considered "good" when building masks for
        cutouts. For example, if pixels in the DQ array can be
        combinations of 1, 2, 4, and 8 flags and one wants to consider DQ
        "defects" having flags 2 and 4 as being acceptable, then ``dqbitmask``
        should be set to 2+4=6. Then a DQ pixel having values 2,4, or 6
        will be considered a good pixel, while a DQ pixel with a value,
        e.g., 1+2=3, 4+8=12, etc. will be flagged as a "bad" pixel.

        Alternatively, one can enter a comma- or '+'-separated list
        of integer bit flags that should be added to obtain the
        final "good" bits. For example, both ``4,8`` and ``4+8``
        are equivalent to setting ``dqbitmask`` to 12.

        | Default value (0) will make *all* non-zero
          pixels in the DQ mask to be considered "bad" pixels, and the
          corresponding image pixels will be flagged in the ``mask`` property
          of the returned cutouts.

        | Set ``dqbitmask`` to `None` to not consider DQ array when computing
          cutout's ``mask``.

        | In order to reverse the meaning of the ``dqbitmask``
          parameter from indicating values of the "good" DQ flags
          to indicating the "bad" DQ flags, prepend '~' to the string
          value. For example, in order to mask only pixels that have
          corresponding DQ flags 4 and 8 and to consider
          as "good" all other pixels set ``dqbitmask`` to ``~4+8``, or ``~4,8``.
          To obtain the same effect with an `int` input value (except for 0),
          enter ``-(4+8+1)=-9``. Following this convention,
          a ``dqbitmask`` string value of ``'~0'`` would be equivalent to
          setting ``dqbitmask=None``.

    imweight: numpy.ndarray, None, optional
        Pixel weight array corresponding to ``imdata``.

    data_units: {'counts', 'rate'}, optional
        Indicates the type of data units: count-like or rate-like (counts per
        unit of time). This provides the information necessary for unit
        conversion when needed.

    exptime: float, optional
        Exposure time of image ``imdata``.

    pad: int, optional
        Number of pixels to pad around the minimal rectangle enclosing
        a source segmentation.

    Returns
    -------
    segments : list of Cutout
        A list of extracted ``Cutout`` s.

    """
    if isinstance(catalog, ImageCatalog):
        catalog = catalog.catalog

    ny, nx = segmentation_image.shape
    pad = _ceil(pad) if pad >=0 else _floor(pad)

    # find IDs present both in the catalog AND segmentation image
    ids, cat_indices, _ = np.intersect1d(
        np.asarray(catalog['id']),
        np.setdiff1d(np.unique(segmentation_image), [0]),
        return_indices=True
    )

    segments = []
    if 'weight' in catalog.colnames:
        src_weights = catalog['weight']
    else:
        src_weights = None

    for sid, sidx in zip(ids, cat_indices):
        # find indices of pixels having a 'sid' ID:
        mask = segmentation_image == sid
        idx = np.where(mask)

        # find the boundary of the segmentation region enlarged by 1 on each
        # side, to be on the safe side when re-projecting the bounding box
        # to input (distorted) images:
        x1 = np.min(idx[1])
        x2 = np.max(idx[1])
        y1 = np.min(idx[0])
        y2 = np.max(idx[0])

        if x1 <= 0 or y1 <= 0 or x2 >= (nx - 1) or y2 >= (ny - 1):
            # skip sources sitting at the edge of segmentation image.
            # we simply do not know if these are "complete" sources or
            # that these sources did not extend beyond the current
            # boundaries of the segmentation image.
            continue

        # apply extra padding:
        x1 -= pad
        x2 += pad
        y1 -= pad
        y2 += pad

        src_pos = (catalog['x'][sidx], catalog['y'][sidx])
        if src_weights is None:
            src_weight = None
        else:
            src_weight = src_weights[sidx]

        cutout = Cutout(imdata, imwcs, blc=(x1, y1), trc=(x2, y2),
                        src_pos=src_pos, src_weight=src_weight,
                        dq=imdq, weight=imweight, src_id=sid,
                        data_units=data_units, exptime=exptime, fillval=0)

        cutout.mask |= np.logical_not(mask[cutout.extraction_slice])

        if imdq is not None:
            cutout.mask |= bitfield_to_boolean_mask(
                cutout.dq, ignore_flags=dqbitmask, good_mask_value=False
            )

        if not np.all(cutout.mask): # ignore cutouts without any good pixels
            segments.append(cutout)

    return segments


def create_input_image_cutouts(primary_cutouts, imdata, imwcs, imdq=None,
                               dqbitmask=0, imweight=None, data_units='counts',
                               exptime=1, pad=1):
    """
    A function for creating cutouts in one image from cutouts from another
    image. Specifically, this function maps input cutouts to quadrilaterals
    in some image and then finds minimal enclosing rectangle that encloses
    the quadrilateral. This minimal rectangle is then padded as requested
    and a new `Cutout` from ``imdata`` is created. If an input ``Cutout``
    from ``primary_cutouts`` does not fit entirely within ``imdata``
    (after padding) that ``Cutout`` is ignored.

    Parameters
    ----------
    primary_cutouts : list of Cutout
        A list of ``Cutout``s that need to be mapped to *another* image.

    imdata: numpy.ndarray
        Image data array to which ``primary_cutouts`` should be mapped.

    imwcs: astropy.wcs.WCS
        World coordinate system of image ``imdata``.

    imdq: numpy.ndarray, None, optional
        Data quality array corresponding to ``imdata``.

    dqbitmask : int, str, None, optional
        Integer sum of all the DQ bit values from the input ``imdq``
        DQ array that should be considered "good" when building masks for
        cutouts. For more details, see `create_primary_cutouts`.

    imweight: numpy.ndarray, None, optional
        Pixel weight array corresponding to ``imdata``.

    data_units: {'counts', 'rate'}, optional
        Indicates the type of data units: count-like or rate-like (counts per
        unit of time). This provides the information necessary for unit
        conversion when needed.

    exptime: float, optional
        Exposure time of image ``imdata``.

    pad: int, optional
        Number of pixels to pad around the minimal rectangle enclosing
        a mapped cutout (a cutout to be extracted).

    Returns
    -------
    imcutouts : list of Cutout
        A list of extracted ``Cutout``s.

    valid_input_cutouts : list of Cutout
        A list of ``Cutout``s from ``primary_cutouts`` that completely fit
        within ``imdata``. There is a one-to-one correspondence
        between cutouts in ``valid_input_cutouts`` and cutouts in
        ``imcutouts``. That is, each ``Cutout`` from ``imcutouts`` has a
        corresponding (at the same position in the list) ``Cutout``
        in ``valid_input_cutouts``.

    """
    imcutouts = []
    valid_input_cutouts = []
    ny, nx = imdata.shape
    for ct in primary_cutouts:
        imfootprint = imwcs.all_world2pix(ct.get_bbox('world'), 0,
                                          accuracy=1e-5, maxiter=50)

        # find a conservative bounding *rectangle*:
        x1 = _floor(imfootprint[:, 0].min() - pad)
        y1 = _floor(imfootprint[:, 1].min() - pad)
        x2 = _ceil(imfootprint[:, 0].max() + pad)
        y2 = _ceil(imfootprint[:, 1].max() + pad)

        # skip a cutout if its bounding rectangle is entirely inside
        # the image's data array:
        if (x1 < 0 or x1 >= nx or y1 < 0 or y1 > ny or
            x2 < 0 or x2 >= nx or y2 < 0 or y2 > ny):
            continue

        try:
            imct = Cutout(imdata, imwcs, blc=(x1, y1), trc=(x2, y2),
                          src_weight=ct.src_weight, dq=imdq, weight=imweight,
                          src_id=ct.src_id, data_units=data_units,
                          exptime=exptime, fillval=0)

            if imdq is not None:
                imct.mask |= bitfield_to_boolean_mask(
                    imct.dq, ignore_flags=dqbitmask, good_mask_value=False
                )

            if np.all(imct.mask):
                continue

        except (NoOverlapError, PartialOverlapError):
            continue

        # only when there is at least partial overlap,
        # compute source position from the primary_cutouts to
        # imcutouts using full WCS transformations:
        imct.cutout_src_pos = imct.world2pix(
            ct.pix2world([ct.cutout_src_pos]))[0].tolist()

        imcutouts.append(imct)
        valid_input_cutouts.append(ct)

    return imcutouts, valid_input_cutouts


def drz_from_input_cutouts(input_cutouts, segmentation_image, imdata, imwcs,
                           imdq=None, dqbitmask=0, imweight=None,
                           data_units='counts', exptime=1, pad=1,
                           combine_seg_mask=True):
    """
    A function for creating cutouts in one image from cutouts from another
    image. Specifically, this function maps input cutouts to quadrilaterals
    in some image and then finds minimal enclosing rectangle that encloses
    the quadrilateral. This minimal rectangle is then padded as requested
    and a new `Cutout` from ``imdata`` is created.

    This function is similar to ``create_input_image_cutouts`` the main
    differences being how partial overlaps are treated and "bad" pixels
    (pixels that are not within the segmentation map) are masked in the
    ``mask`` attribute.

    If an input ``Cutout`` from ``input_cutouts`` does not fit even partially
    within ``imdata`` (after padding) that ``Cutout`` is ignored.
    If an input ``Cutout`` from ``input_cutouts`` does fit partially
    within ``imdata`` (after padding) that ``Cutout`` is filled with zeros.

    Parameters
    ----------
    input_cutouts : list of Cutout
        A list of ``Cutout``s that need to be mapped to *another* image.

    segmentation_image: numpy.ndarray
        A 2D segmentation image identifying sources from the catalog
        in ``imdata``. This is used for creating boolean mask of ``bad``
        (not within a segmentation region) pixels.

    imdata: numpy.ndarray
        Image data array to which ``input_cutouts`` should be mapped.

    imwcs: astropy.wcs.WCS
        World coordinate system of image ``imdata``.

    imdq: numpy.ndarray, None, optional
        Data quality array corresponding to ``imdata``.

    dqbitmask : int, str, None, optional
        Integer sum of all the DQ bit values from the input ``imdq``
        DQ array that should be considered "good" when building masks for
        cutouts. For more details, see `create_primary_cutouts`.

    imweight: numpy.ndarray, None, optional
        Pixel weight array corresponding to ``imdata``.

    data_units: {'counts', 'rate'}, optional
        Indicates the type of data units: count-like or rate-like (counts per
        unit of time). This provides the information necessary for unit
        conversion when needed.

    exptime: float, optional
        Exposure time of image ``imdata``.

    pad: int, optional
        Number of pixels to pad around the minimal rectangle enclosing
        a mapped cutout (a cutout to be extracted).

    combine_seg_mask: bool, optional
        Indicates whether to combine segmanetation mask with cutout's
        mask. When `True`, segmentation image is used to create a mask that
        indicates "good" pixels in the image. This mask is combined with
        cutout's mask.

    Returns
    -------
    imcutouts : list of Cutout
        A list of extracted ``Cutout``s.

    valid_input_cutouts : list of Cutout
        A list of ``Cutout``s from ``primary_cutouts`` that at least partially
        fit within ``imdata``. There is a one-to-one correspondence
        between cutouts in ``valid_input_cutouts`` and cutouts in
        ``imcutouts``. That is, each ``Cutout`` from ``imcutouts`` has a
        corresponding (at the same position in the list) ``Cutout``
        in ``valid_input_cutouts``.

    """
    imcutouts = []
    valid_input_cutouts = []
    ny, nx = imdata.shape
    pad = _ceil(pad) if pad >=0 else _floor(pad)

    for ct in input_cutouts:
        imfootprint = imwcs.all_world2pix(ct.get_bbox('world'), 0,
                                          accuracy=1e-5, maxiter=50)

        # find a conservative bounding *rectangle*:
        x1 = _floor(imfootprint[:, 0].min() - pad)
        y1 = _floor(imfootprint[:, 1].min() - pad)
        x2 = _ceil(imfootprint[:, 0].max() + pad)
        y2 = _ceil(imfootprint[:, 1].max() + pad)

        # skip a cutout if its bounding rectangle is entirely inside
        # the image's data array:
        try:
            imct = Cutout(imdata, imwcs, blc=(x1, y1), trc=(x2, y2),
                          src_weight=ct.src_weight, dq=imdq, weight=imweight,
                          src_id=ct.src_id, data_units=data_units,
                          exptime=exptime, mode='fill', fillval=0)

        except NoOverlapError:
            continue

        # update cutout mask with segmentation image:
        if combine_seg_mask:
            seg = np.zeros_like(imct.data)
            seg[imct.insertion_slice] = segmentation_image[imct.extraction_slice]
            imct.mask |= ~(seg == ct.src_id)

        if imdq is not None:
            imct.mask |= bitfield_to_boolean_mask(
                imct.dq, ignore_flags=dqbitmask, good_mask_value=False
            )

        if np.all(imct.mask):
            continue

        # only when there is at least partial overlap,
        # compute source position from the primary_cutouts to
        # imcutouts using full WCS transformations:
        imct.cutout_src_pos = imct.world2pix(
            ct.pix2world([ct.cutout_src_pos]))[0].tolist()

        imcutouts.append(imct)
        valid_input_cutouts.append(ct)

    return imcutouts, valid_input_cutouts


def create_cutouts(primary_cutouts, segmentation_image,
                   drz_data, drz_wcs, flt_data, flt_wcs,
                   drz_dq=None, drz_dqbitmask=0, drz_weight=None,
                   drz_data_units='rate', drz_exptime=1,
                   flt_dq=None, flt_dqbitmask=0, flt_weight=None,
                   flt_data_units='counts', flt_exptime=1,
                   pad=2, combine_seg_mask=True):
    """
    A function for mapping "primary cutouts" (cutouts formed form a
    drizzle-combined image) to "input images" (generally speaking, distorted
    images) and some other "drizzle-combined" image. This "other"
    drizzle-combined image may be the same image used to create primary
    cutouts.

    This function performs the following mapping/cutout extractions:

    > ``primary_cutouts`` -> ``imcutouts`` -> ``drz_cutouts``

    That is, this function takes as input ``primary_cutouts`` and finds
    equivalent cutouts in the "input" (distorted) "flt" image. Then it takes
    the newly found ``imcutouts`` cutouts and finds/extracts equivalent
    cutouts in the "drz" (usually distortion-corrected) image. Fundamentally,
    this function first calls `create_input_image_cutouts` to create
    ``imcutouts`` and then it calls `drz_from_input_cutouts` to create
    ``drz_cutouts`` .

    Parameters
    ----------
    primary_cutouts : list of Cutout
        A list of ``Cutout`` s that need to be mapped to *another* image.

    segmentation_image: numpy.ndarray
        A 2D segmentation image identifying sources from the catalog
        in ``imdata``. This is used for creating boolean mask of ``bad``
        (not within a segmentation region) pixels.

    drz_data: numpy.ndarray
        Image data array of "drizzle-combined" image.

    drz_wcs: astropy.wcs.WCS
        World coordinate system of "drizzle-combined" image.

    flt_data: numpy.ndarray
        Image data array of "distorted" image (input to the drizzle).

    flt_wcs: astropy.wcs.WCS
        World coordinate system of "distorted" image.

    drz_dq: numpy.ndarray, None, optional
        Data quality array corresponding to ``drz_data``.

    drz_dqbitmask : int, str, None, optional
        Integer sum of all the DQ bit values from the input ``drz_dq``
        DQ array that should be considered "good" when building masks for
        cutouts. For more details, see `create_primary_cutouts`.

    drz_weight: numpy.ndarray, None, optional
        Pixel weight array corresponding to ``drz_data``.

    drz_data_units: {'counts', 'rate'}, optional
        Indicates the type of data units for the ``drz_data`` :
        count-like or rate-like (counts per unit of time).
        This provides the information necessary for unit
        conversion when needed.

    drz_exptime: float, optional
        Exposure time of image ``drz_data``.

    flt_dq: numpy.ndarray, None, optional
        Data quality array corresponding to ``flt_data``.

    flt_dqbitmask : int, str, None, optional
        Integer sum of all the DQ bit values from the input ``flt_dq``
        DQ array that should be considered "good" when building masks for
        cutouts. For more details, see `create_primary_cutouts`.

    flt_weight: numpy.ndarray, None, optional
        Pixel weight array corresponding to ``flt_data``.

    flt_data_units: {'counts', 'rate'}, optional
        Indicates the type of data units for the ``flt_data``:
        count-like or rate-like (counts per unit of time).
        This provides the information necessary for unit
        conversion when needed.

    flt_exptime: float, optional
        Exposure time of image ``flt_data``.

    pad: int, optional
        Number of pixels to pad around the minimal rectangle enclosing
        a mapped cutout (a cutout to be extracted).

    combine_seg_mask: bool, optional
        Indicates whether to combine segmanetation mask with cutout's
        mask. When `True`, segmentation image is used to create a mask that
        indicates "good" pixels in the image. This mask is combined with
        cutout's mask.

    Returns
    -------
    flt_cutouts : list of Cutout
        A list of ``Cutout`` s extracted from the ``flt_data``. These cutouts
        are large enough to enclose cutouts from the input
        ``primary_cutouts`` when ``pad=1`` (to make sure even partial pixels
        are included).

    drz_cutouts : list of Cutout
        A list of extracted ``Cutout`` s from the ``drz_data``. These cutouts
        are large enough to enclose cutouts from the
        ``flt_cutouts`` when ``pad=1`` (to make sure even partial pixels
        are included).

    """
    # map initial cutouts to FLT image:
    imcutouts1, drz_cutouts1 = create_input_image_cutouts(
        primary_cutouts=primary_cutouts,
        imdata=flt_data,
        imwcs=flt_wcs,
        imdq=flt_dq,
        dqbitmask=flt_dqbitmask,
        imweight=flt_weight,
        data_units=flt_data_units,
        exptime=flt_exptime,
        pad=pad # add two pixels in order to confidently allow 1/2 pixel shifts
    )

    if len(imcutouts1) == 0:
        return [], []

    pix_scale_ratio = imcutouts1[0].pscale / drz_cutouts1[0].pscale

    # map FLT cutouts back to drizzled image:
    drz_cutouts, flt_cutouts = drz_from_input_cutouts(
        input_cutouts=imcutouts1,
        segmentation_image=segmentation_image,
        imdata=drz_data,
        imwcs=drz_wcs,
        imdq=drz_dq,
        dqbitmask=drz_dqbitmask,
        imweight=drz_weight,
        data_units=drz_data_units,
        exptime=drz_exptime,
        pad=pad * pix_scale_ratio,
        combine_seg_mask=combine_seg_mask
    )

    return flt_cutouts, drz_cutouts


class Cutout(object):
    """
    This is a class designed to facilitate work with image cutouts. It holds
    both information about the cutout (location in the image) as well as
    information about the image and source: source ID, exposure time, image
    units, ``WCS``, etc.

    This class also provides convinience tools for creating cutouts,
    saving them to or loading from files, and for
    converting pixel coordinates to world coordinates (and vice versa) using
    cutout's pixel grid while preserving all distortion corrections
    from image's ``WCS``.

    Parameters
    ----------
    data: numpy.ndarray
        Image data from which the cutout will be extracted.

    wcs: astropy.wcs.WCS
        World Coordinate System object describing coordinate transformations
        from image's pixel coordinates to world coordinates.

    blc: tuple of two int
        Bottom-Left Corner coordinates ``(x, y)`` in the ``data`` of the cutout
        to be extracted.

    trc: tuple of two int, None, optional
        Top-Right Corner coordinates ``(x, y)`` in the ``data`` of the cutout
        to be extracted. Pixel with the coordinates ``trc`` is included.
        When ``trc`` is set to `None`, ``trc`` is set to the shape of the
        ``data`` image: ``(nx, ny)``.

    src_pos: tuple of two int, None, optional
        Image coordinates ``(x, y)`` **in the input ``data`` image**
        of the source contained in this cutout. If ``src_pos``
        is set to the default value (`None`), then it will be set to the
        center of the cutout.

        .. warning::

           **TODO:** The algorithm for ``src_pos`` computation
           most likely will need to be revised to obtain better estimates
           for the position of the source in the cutout.

    src_weight : float, None, optional
        The weight of the source in the cutout to be used in alignment when
        fitting geometric transformations.

    dq: numpy.ndarray
        Data quality array associated with image data. If provided, this
        array will be cropped the same way as image data and stored within
        the ``Cutout`` object.

    weight: numpy.ndarray
        Weight array associated with image data. If provided, this
        array will be cropped the same way as image data and stored within
        the ``Cutout`` object.

    src_id : any type, None
        Anything that can be used to associate the source being extracted
        with a record in a catalog. This value is simply stored within the
        ``Catalog`` object.

    data_units: {'counts', 'rate'}, optional
        Indicates the type of data units: count-like or rate-like (counts per
        unit of time). This provides the information necessary for unit
        conversion when needed.

    exptime: float, optional
        Exposure time of image ``imdata``.

    mode: {'strict', 'fill'}
        Allowed overlap between extraction rectangle for the cutout and the
        input image. When ``mode`` is ``'strict'`` then a `PartialOverlapError`
        error will be raised if the extraction rectangle is not *completely*
        within the boundaries of input image. When ``mode`` is ``'fill'``,
        then parts of the cutout that are outside the boundaries of the
        input image will be filled with the value specified by the
        ``fillval`` parameter.

    fillval: scalar
        All elements of the cutout that are outside the input image will be
        assigned this value. This parameter is ignored when ``mode`` is
        set to ``'strict'``.

    Raises
    ------
    `NoOverlapError`
        When cutout is completely outside of the input image.

    `PartialOverlapError`
        When cutout only partially overlaps input image and ``mode`` is set to
        ``'strict'``.

    """
    DEFAULT_ACCURACY = 1.0e-5
    DEFAULT_MAXITER = 50
    DEFAULT_QUIET = True

    def __init__(self, data, wcs, blc=(0, 0), trc=None,
                 src_pos=None, src_weight=None, dq=None, weight=None, src_id=0,
                 data_units='rate', exptime=1, mode='strict', fillval=np.nan):
        if trc is None and data is None:
            raise ValueError("'trc' cannot be None when 'data' is None.")

        if data is None:
            nx = trc[0] + 1
            ny = trc[1] + 1
            data_dtype = np.float32
        else:
            ny, nx = data.shape
            data_dtype = data.dtype

        if mode not in ['strict', 'fill']:
            raise ValueError("Argument 'mode' must be either 'strict' or "
                             "'fill'.")

        self._onx = nx
        self._ony = ny

        if trc is None:
            trc = (nx - 1, ny - 1)

        if blc[0] >= nx or blc[1] >= ny or trc[0] < 0 or trc[1] < 0:
            raise NoOverlapError(
                "Cutout's extraction box does not overlap image data array."
            )

        else:
            if trc[0] < blc[0] or trc[1] < blc[1]:
                raise ValueError("Ill-formed extraction box: coordinates of "
                                 "the top-right corner cannot be smaller than "
                                 "the coordinates of the bottom-left corner.")

            if mode == 'strict' and (blc[0] < 0 or blc[1] < 0 or
                                     trc[0] >= nx or trc[1] >= ny):
                raise PartialOverlapError(
                    "Cutout's extraction box only partially overlaps image "
                    "data array."
                )

        self._blc = (blc[0], blc[1])
        self._trc = (trc[0], trc[1])
        self.src_pos = src_pos
        self.src_weight = src_weight

        # create data and mask arrays:
        cutout_data = np.full((self.height, self.width), fill_value=fillval,
                              dtype=data_dtype)
        self._mask = np.ones_like(cutout_data, dtype=np.bool_)

        # find overlap bounding box:
        bbx1 = max(0, blc[0])
        bby1 = max(0, blc[1])
        bbx2 = min(nx - 1, trc[0]) + 1
        bby2 = min(ny - 1, trc[1]) + 1
        extract_slice = np.s_[bby1:bby2, bbx1:bbx2]
        insert_slice = np.s_[bby1-blc[1]:bby2-blc[1], bbx1-blc[0]:bbx2-blc[0]]

        # get data and fill the mask
        if data is not None:
            cutout_data[insert_slice] = data[extract_slice]
        self._mask[insert_slice] = False

        # flag "bad" (NaN, inf) pixels:
        self._mask |= np.logical_not(np.isfinite(cutout_data))

        # get DQ array if provided:
        if dq is None:
            self._dq = None

        else:
            if dq.shape != (ny, nx):
                raise ValueError("Image's DQ array shape must match the shape "
                                 "of image 'data'.")

            self._dq = np.zeros_like(cutout_data, dtype=dq.dtype)
            self._dq[insert_slice] = dq[extract_slice]

        # get weights array if provided:
        if weight is None:
            self._weight = None

        else:
            if weight.shape != (ny, nx):
                raise ValueError("Image's weight array shape must match the "
                                 "shape of image 'data'.")

            self._weight = np.zeros_like(cutout_data, dtype=weight.dtype)
            self._weight[insert_slice] = weight[extract_slice]

        self._src_id = src_id

        self._dx = 0
        self._dy = 0

        self._data = cutout_data

        self._wcs = wcs
        self._naxis = [i for i in cutout_data.shape[::-1]]
        for k, naxis_k in enumerate(self._naxis):
            self.__dict__['naxis{:d}'.format(k + 1)] = naxis_k

        self._eslice = extract_slice
        self._islice = insert_slice

        self.exptime = exptime
        self.data_units = data_units

    @property
    def src_pos(self):
        """ Get/set source position in the *cutout's image*. """
        return self._src_pos

    @src_pos.setter
    def src_pos(self, src_pos):
        """ Get/set source position in the *cutout's image*. """
        x1, y1 = self.blc
        if src_pos is None:
            x2, y2 = self.trc
            self._src_pos = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
        else:
            self._src_pos = tuple(src_pos)[:2]

    @property
    def src_weight(self):
        """ Get/set source's weight for fitting geometric transformations. """
        return self._src_weight

    @src_weight.setter
    def src_weight(self, src_weight):
        if src_weight is not None and np.any(src_weight < 0.0):
            raise ValueError("Source weight must be a non-negative number "
                             "or None.")
        self._src_weight = src_weight

    @property
    def cutout_src_pos(self):
        """ Get/set source position in the *cutout's image coordinates*. """
        x1, y1 = self.blc
        cx, cy = self._src_pos
        return (cx - x1, cy - y1)

    @cutout_src_pos.setter
    def cutout_src_pos(self, src_pos):
        """ Get/set source position in the *cutout's image coordinates*. """
        x1, y1 = self.blc
        if src_pos is None:
            x2, y2 = self.trc
            self._src_pos = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
        else:
            self._src_pos = (src_pos[0] + x1, src_pos[1] + y1)

    @property
    def exptime(self):
        """ Get/Set exposure time. """
        return self._exptime

    @exptime.setter
    def exptime(self, exptime):
        if exptime <= 0:
            raise ValueError("'exptime' must be positive.")
        self._exptime = exptime

    @property
    def data_units(self):
        """ Get/Set image data units. Possible values are:
        'rate' or 'counts'.

        """
        return self._data_units

    @data_units.setter
    def data_units(self, units):
        units = units.lower()
        if units not in ['rate', 'counts']:
            raise ValueError("Allowed image data units are: 'rate' or "
                             "'counts'.")
        self._data_units = units

    @property
    def naxis(self):
        """ Get FITS ``NAXIS`` property of the cutout. """
        return self._naxis

    @property
    def extraction_slice(self):
        """
        Get slice object that shows the slice *in the input data array*
        used to extract the cutout.

        """
        return self._eslice

    @property
    def insertion_slice(self):
        """
        Get slice object that shows the slice *in the cutout data array*
        into which image data were placed. This slice coincides with  the
        entire cutout data array when ``mode`` is ``'strict'`` but can point
        to a smaller region when ``mode`` is ``'fill'``.

        """
        return self._islice

    @property
    def src_id(self):
        """ Set/Get source ID. """
        return self._src_id

    @src_id.setter
    def src_id(self, src_id):
        self._src_id = src_id

    @property
    def data(self):
        """ Get image data. """
        return self._data

    @data.setter
    def data(self, data):
        if data is None:
            raise ValueError("'data' cannot be None.")

        elif data is self._data:
            return

        data = np.array(data)
        if self._data.shape != data.shape:
            raise ValueError(
                "ValueError: could not broadcast input array from shape "
                "({:s}) into shape ({:s})"
                .format(','.join(map(str, data.shape)),
                        ','.join(map(str, self._data.shape)))
            )
        self._data = data

    @property
    def mask(self):
        """ Set/Get cutout's mask. """
        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask is None:
            raise ValueError("'mask' cannot be None.")
        elif mask is not self._mask:
            mask = np.array(mask, dtype=np.bool_)
            if self._mask.shape != mask.shape:
                raise ValueError(
                    "ValueError: could not broadcast input array from shape "
                    "({:s}) into shape ({:s})"
                    .format(','.join(map(str, mask.shape)),
                            ','.join(map(str, self._mask.shape)))
                )
            self._mask = mask

    @property
    def dq(self):
        """ Set/Get cutout's data quality. """
        return self._dq

    @dq.setter
    def dq(self, dq):
        if dq is None:
            self._dq = None

        elif dq is not self._dq:
            dq = np.array(dq)
            if dq.shape != self._data.shape:
                raise ValueError("Cutout's DQ array shape must match the "
                                 "shape of cutout's image_data.")
            self._dq = dq

    @property
    def weight(self):
        """ Set/Get cutout's pixel weight. """
        return self._weight

    @weight.setter
    def weight(self, weight):
        if weight is None:
            self._weight = None

        elif weight is not self._weight:
            weight = np.array(weight)
            if weight.shape != self._data.shape:
                raise ValueError("Cutout's weight array shape must match the "
                                 "shape of cutout's image_data.")
            self._weight = weight

    @property
    def blc(self):
        """ Set/Get coordinate of the bottom-left corner. """
        return self._blc

    @blc.setter
    def blc(self, blc):
        self._blc = (blc[0], blc[1])

    @property
    def trc(self):
        """ Set/Get coordinate of the top-right corner. """
        return self._trc

    @trc.setter
    def trc(self, trc):
        self._trc = (trc[0], trc[1])

    @property
    def dx(self):
        """
        Set/Get displacement of the image grid along the ``X``-axis in pixels.

        """
        return self._dx

    @dx.setter
    def dx(self, dx):
        self._dx = dx

    @property
    def dy(self):
        """
        Set/Get displacement of the image grid along the ``Y``-axis in pixels.

        """
        return self._dy

    @dy.setter
    def dy(self, dy):
        self._dy = dy

    @property
    def width(self):
        """ Get width of the cutout. """
        return self._trc[0] - self._blc[0] + 1

    @property
    def height(self):
        """ Get width of the cutout. """
        return self._trc[1] - self._blc[1] + 1

    def get_bbox(self, wrt='orig'):
        """ Get a `numpy.ndarray` of pixel coordinates of vertices of
        the bounding box. The returned array has the shape `(4, 2)` and
        contains the coordinates of the outer corners of pixels (centers of
        pixels are considered to have integer coordinates).

        Parameters
        ----------
        wrt : {'orig', 'blc', 'world'}, optional

        """
        if wrt not in ['orig', 'blc', 'world']:
            raise ValueError("'wrt' must be one of the following: "
                             "'orig', 'blc', 'world'.")

        if wrt == 'orig':
            x1, y1 = self._blc
            x2, y2 = self._trc

        else:
            x1 = 0
            y1 = 0
            x2 = self._trc[0] - self._blc[0]
            y2 = self._trc[1] - self._blc[1]

        bbox = np.asarray(
            [[x1 - 0.5, y1 - 0.5],
             [x1 - 0.5, y2 + 0.5],
             [x2 + 0.5, y2 + 0.5],
             [x2 + 0.5, y1 - 0.5]]
        )

        if wrt == 'world':
            bbox = self.pix2world(bbox)

        return bbox

    def world2pix(self, *args, origin=0):
        """ Convert world coordinates to _cutout_'s pixel coordinates. """
        nargs = len(args)

        if nargs == 2:
            try:
                ra = np.asarray(args[0], dtype=np.float64)
                dec = np.asarray(args[1], dtype=np.float64)
                vect1D = True
            except:
                raise TypeError("When providing two arguments, they must "
                                "be (x, y) where x and y are Nx1 vectors.")

        elif nargs == 1:
            try:
                rd = np.asarray(args[0], dtype=np.float64)
                ra = rd[:, 0]
                dec = rd[:, 1]
                vect1D = False
            except:
                raise TypeError("When providing one argument, it must be an "
                                "array of shape Nx2 containing Ra & Dec.")

        else:
            raise TypeError("Expected 2 or 3 arguments, {:d} given."
                            .format(nargs))

        x, y = self._wcs.all_world2pix(
            ra, dec, origin,
            accuracy=self.DEFAULT_ACCURACY,
            maxiter=self.DEFAULT_MAXITER,
            quiet=self.DEFAULT_QUIET
        )
        x -= (self._blc[0] - self._dx)
        y -= (self._blc[1] - self._dy)

        if vect1D:
            return [x, y]
        else:
            return np.dstack([x, y])[0]

    def pix2world(self, *args, origin=0):
        """ Convert _cutout_'s pixel coordinates to world coordinates. """
        nargs = len(args)

        if nargs == 2:
            try:
                x = np.asarray(args[0], dtype=np.float64)
                y = np.asarray(args[1], dtype=np.float64)
                vect1D = True
            except:
                raise TypeError("When providing two arguments, they must "
                                "be (x, y) where x and y are Nx1 vectors.")

        elif nargs == 1:
            try:
                xy = np.asarray(args[0], dtype=np.float64)
                x = xy[:, 0]
                y = xy[:, 1]
                vect1D = False
            except:
                raise TypeError("When providing one argument, it must be an "
                                "array of shape Nx2 containing x & y.")

        else:
            raise TypeError("Expected 2 or 3 arguments, {:d} given."
                            .format(nargs))

        x += (self._blc[0] - self._dx)
        y += (self._blc[1] - self._dy)
        ra, dec = self._wcs.all_pix2world(x, y, origin)

        if vect1D:
            return [ra, dec]
        else:
            return np.dstack([ra, dec])[0]

    @property
    def wcs(self):
        """ Get image's WCS from which the cutout was extracted. """
        return self._wcs

    @property
    def pscale(self):
        """ Get pixel scale in the tangent plane at the reference point. """
        if self._wcs is None:
            raise ValueError("WCS was not set. Unable to compute pixel scale.")
        return np.sqrt(fitswcs.utils.proj_plane_pixel_area(self._wcs))
