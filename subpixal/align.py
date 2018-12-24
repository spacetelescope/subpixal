"""
Main module that performs image alignment and WCS correction.

:Author: Mihai Cara (for help, contact `HST Help Desk <https://hsthelp.stsci.edu>`_)

:License: :doc:`../LICENSE`

"""
import collections
import logging
import numbers
import copy
import tempfile

import numpy as np
from astropy.io import fits
from astropy import wcs
from stwcs.wcsutil import HSTWCS

from drizzlepac import linearfit, updatehdr

from . import cutout
from .blot import blot_cutout
from . import cc
from .utils import parse_file_name, _create_tmp_fits_file


__all__ = ['align_images', 'find_linear_fit', 'correct_wcs',
           'update_image_wcs']


log = logging.getLogger(__name__)
log.setLevel(logging.NOTSET)


_ASTROPY_WCS_DISTORTIONS = ['cpdis1', 'cpdis2', 'det2im1', 'det2im2', 'sip']

_SUPPORTED_FITGEOM = ['shift', 'rscale', 'general']


def _create_tmp_reference_file(image_file):
    tmpf = None
    try:
        hdulist = fits.open(image_file)

        tmpf = tempfile.NamedTemporaryFile(
            mode='wb', suffix='.fits', prefix='tmp_refimage_', dir='./',
            delete=True,
        )

        hdulist.writeto(tmpf)
        tmpf.file.flush()
        tmpf.file.seek(0)
    except:
        if tmpf is not None:
            tmpf.close()
        raise

    return tmpf


def align_images(catalog, resample, wcslin=None, fitgeom='general',
                 nclip=3, sigma=3.0, nmax=10, eps_shift=3e-3,
                 wcsname='SUBPIXAL', iterative=True, history='last'):
    """
    Perform *relative* image alignment using sub-pixel cross-correlation.
    Image alignment is performed by adusting each image's WCS so that images
    align on the sky (i.e., sources from the catalog overlap). Input image
    data (provided through the ``resample`` parameter) are not changed.

    Parameters
    ----------
    catalog : catalogs.ImageCatalog
        A catalog object of `~catalogs.ImageCatalog`-derived type. This object
        will hold source-finding and source filtering parameters and should
        be able to find sources in provided images on demand.

    resample : resample.Resample
        An object of `resample.Resample`-derived type that can resample its
        images onto a common output grid.

    wcslin : astropy.wcs.WCS, None, optional
        A `~astropy.wcs.WCS` object that does not have non-linear distortions.
        This WCS defines a tangen plane in which image alignemnt will be
        performed. When not provided or set to `None`,
        it is set to ``drz_cutouts[0].wcs``.

    fitgeom : {'shift', 'rscale', 'general'}, optional
        The fitting geometry to be used in fitting cutout displacements.
        This parameter is used in fitting the offsets, rotations
        and/or scale changes from the matched object lists. The 'general'
        fit geometry allows for independent scale and rotation for
        each axis.

    nclip : int, optional
        Number (a non-negative integer) of clipping iterations in fit.

    sigma : float, optional
        Clipping limit in sigma units.

    nmax : int, tuple of two int, optional
        A positive integer number indicating the number of resample-alignment
        iterations to be performed. After detecting that resampled images
        do not change significantly, the algorithm will automatically switch
        to a faster resampling :py:meth:`~resample.Resample.fast_add_image`
        and :py:meth:`~resample.Resample.fast_drop_image` methods instead of
        performing "full" resample that includes sky re-computation, cosmic
        ray detection, etc.

        When ``nmax`` is a tuple of integers, first number indicates the
        maximum number of iterations to be performed and the second number
        indicates the maximum number of iterations with "full" resample to be
        performed.

    eps_shift : float, optional
        The algorithm will stop iterations when found shifts are below
        ``eps_shift`` value for all images.

    wcsname : str, None, optional
        Label to give newly updated WCS. The default value will set the
        WCS name to `SUBPIXAL`.

    iterative : bool, optional
        If `True`, after each iteration user will be asked whether to
        continue or stop alignment process.

    history : {'all', 'last', None}
        On return this function returns "fit history" containing information
        that can be used to analyze the goodness of fit. When ``history``
        is ``'all'``, then info from each iteration is saved. When ``history``
        is ``'last'`` only info for the last iteration is saved, and when
        ``history`` is `None`, no informationn is saved.

    Returns
    -------
    fit_history : list of dict
        A list of Python dictionaries containing fit information as well as
        "image" information such as image cutouts, blots, cross-correlation
        image, etc.

    """
    if history not in [None, 'last', 'all']:
        raise ValueError("Parameter 'history' must be either None, 'all', or "
                         "'last'.")

    if fitgeom.lower() not in _SUPPORTED_FITGEOM:
        raise ValueError("'fitgeom' must be either '{:s}', or '{:s}'"
                         .format('\', \''.join(_SUPPORTED_FITGEOM[:-1]),
                                 _SUPPORTED_FITGEOM[-1]))

    if isinstance(nmax, (tuple, list)):
        if (len(nmax) != 2
            or (not isinstance(nmax[0], numbers.Integral) or nmax[0] < 1)
            or (not isinstance(nmax[1], numbers.Integral) or nmax[1] < 1)):
            raise ValueError("Parameter 'nmax' must be a positive integer or "
                             "a tuple of two positive integers.")

        nmax_iter, nmax_fr = nmax  # nmax_fr <- nmax of full resamples
        if nmax_fr > nmax_iter:
            raise ValueError("Number of \"full\" resamples cannot be "
                             "larger than the total number of iterations.")

    else:
        if (not isinstance(nmax, numbers.Integral) or nmax < 1):
            raise ValueError("Parameter 'nmax' must be a positive integer or "
                             "a tuple of two positive integers.")

        nmax_iter = nmax
        nmax_fr = None  # maximum number of full resamples

    if not isinstance(eps_shift, numbers.Real) or eps_shift <= 0.0:
        raise ValueError("Parameter 'eps_shift' must be a positive number.")

    nfr = 0  # number of full resamples
    achieved_desired_eps = False
    resample = copy.deepcopy(resample)  # avoid changing input objects
    tmp_ref = None

    # record-keeping
    fit_history = []
    summary = []

    for k in range(nmax_iter):
        iterno = k + 1
        if history:
            run_info = {
                'iteration': iterno,
                'full_resample': False,  # Complete drizzle with median, CR,
                                         # skysub, etc.
                'finfo': []  # file-specific alignment info
            }

        print("\n==============\nITERATION #{:d}\n==============\n"
              .format(iterno))

        if nfr == 0 or nmax_fr is None or nfr < nmax_fr:
            resample.execute()
            if history:
                run_info['full_resample'] = True

            if nfr == 0:
                # fix reference image if not provided (only first time):
                refim = resample.reference_image
                if refim is None or (isinstance(refim, str) and
                                     not refim.strip()):
                    computed_sky = resample.computed_sky
                    tmp_ref = _create_tmp_fits_file(
                        parse_file_name(resample.output_sci)[0],
                        prefix='tmp_refimage_'
                    )
                    resample.reference_image = tmp_ref.name
                    resample.computed_sky = computed_sky

            elif nmax_fr is None and nfr > 0:
                # see if we need other "full" (with CR-detection,
                # sky subtraction, etc.) resample steps:
                fsci_name = parse_file_name(resample.output_sci)[0]
                fdata = fits.getdata(fsci_name)

                qsci_name = parse_file_name(qresample.output_sci)[0]
                qdata = fits.getdata(qsci_name)

                fi = np.finfo(fdata.dtype)
                if np.allclose(qdata, fdata, rtol=2.0 * fi.eps,
                               atol=10.0 * fi.tiny, equal_nan=True):
                    nmax_fr = nfr

            nfr += 1

            qresample = copy.deepcopy(resample)

            # find sources in the drizzled image:
            catalog.set_image(resample.output_sci)
            sources = catalog.catalog

            print("\nCATALOG LENGTH: {:d}".format(len(catalog.catalog)))

            # get segmentation image:
            seg = catalog.get_segmentation_image()

        # align:
        shift_corr = []
        add_file_name = None
        fit_summary = []

        for crclean_image, (imfile, exts) in zip(resample.output_crclean,
                                                resample.input_image_names):

            msg = "*  Aligning image \"{:s}{}\"  *".format(imfile, exts)
            print("\n{0:s}\n{1:s}\n{0:s}\n".format(len(msg) * '*', msg))

            image_sky = resample.computed_sky[imfile]
            qresample.fast_replace_image(drop_file_name=imfile,
                                         add_file_name=add_file_name)

            add_file_name = imfile

            # see if we can create a "DQ"-like mask for drizzled image:
            whtdq = None
            if qresample.output_wht is not None:
                fn, ext = parse_file_name(qresample.output_wht)
                with fits.open(fn) as h:
                    whtdq = np.equal(h[ext].data, 0.0).astype(dtype=np.int)

            fn, ext = parse_file_name(qresample.output_sci)
            with fits.open(fn) as h:
                drz_wcs = wcs.WCS(h[ext], h)
                drz_exptime = h[0].header['EXPTIME']
                drz_units = h[ext].header['BUNIT']
                drz_data = h[ext].data

            prim_cutouts = cutout.create_primary_cutouts(
                sources, seg, drz_data, drz_wcs, imdq=whtdq,
                data_units=('rate' if '/' in drz_units else 'counts'),
                exptime=drz_exptime
            )

            fit, img_info = _align_1image(
                qresample,
                crclean_image,
                image_ext=exts,
                primary_cutouts=prim_cutouts,
                image_sky=image_sky,
                seg=seg,
                nclip=nclip,
                sigma=sigma,
                fitgeom=fitgeom
            )

            fit_summary.append(
                (imfile, fit['resids'].shape[0], fit['rms'],
                 fit['offset'], fit['rot'], fit['rotxy'], fit['scale'])
            )

            if history:
                run_info['finfo'].append(
                    {'image': imfile, 'fit_info': fit, 'image_info': img_info}
                )

            shift_corr.append(np.linalg.norm(fit['offset']))

            # update WCS in image headers:
            with fits.open(imfile, mode='update') as h:
                for e, _, new_wcs in img_info['wcs_info']:
                    print("\nUpdating with new WCS:\n{}".format(new_wcs))
                    update_image_wcs(h, e, new_wcs, wcsname=wcsname)

            with fits.open(crclean_image, mode='update') as h:
                for e, _, new_wcs in img_info['wcs_info']:
                    update_image_wcs(h, e, new_wcs, wcsname=wcsname)

        summary.append((iterno, fit_summary))

        if history == 'all':
            fit_history.append(run_info)
        elif history == 'last':
            fit_history = [run_info]

        if max(shift_corr) < eps_shift:
            achieved_desired_eps = True
            break

        if nmax_fr is None: # or nfr >= nmax_fr:
            qresample.fast_replace_image(drop_file_name=None,
                                         add_file_name=add_file_name)

        if nmax_fr is None:
            # see if we need other "full" (with CR-detection, sky subtraction,
            # etc.) resample steps:
            qsci_name = parse_file_name(qresample.output_sci)[0]
            qdata = fits.getdata(qsci_name)

        if iterative:
            s = input("Do you want to continue? (y/n)\n")
            if s.upper().startswith('N'):
                break

    if achieved_desired_eps:
        print("\nDesired alignment accuracy has been achieved after {:d} iterations.".format(iterno))
    else:
        print("\nThe iterative alignment process has been stopped\n"
              "after reaching maximum number of iterations.")

    fh = open('final_summary.txt', 'w')

    print("\n===============================================")
    print("SUMMARY")
    print("\n===============================================\n")
    for imno in range(len(resample.input_image_names)):
        line = "\n-------------------------\nFILE NAME: {}\n\n" \
               .format(resample.input_image_names[imno][0])
        print(line)
        fh.write(line + '\n')
        fh.flush()

        for k, fi in summary:
            (imfile, nmatch, rms, offset, rot, rotxy, scale) = fi[imno]
            line = ("{:2d}: nmatch={:3d}, xrms={:5.2g}, yrms={:5.2g},  "
                    "dx={:8.4f}, dy={:8.4f},  rot={:8.5f},  sx={:10.7f}, "
                    "sy={:10.7f}"
                    .format(k, nmatch, *rms, *offset, rot, *scale))
            print(line)
            fh.write(line + '\n')
            fh.flush()
    fh.close()

    return fit_history


def _align_1image(resample, image, image_ext, primary_cutouts, seg,
                 image_sky=None, wcslin=None,
                 fitgeom='general', nclip=3, sigma=3.0):
    # resample.fast_drop_image(image)

    img_info = {
        'file_name': image,
        'wcs_info': [],  # a list of: [extension, original WCS, corrected WCS]
        'fits_ext': [],  # image ext. from which an image cutout was extracted
        'image_cutouts': [],
        'driz_cutouts': [],
        'blotted_cutouts': [],  # non-shifted blot images of drizzled cutouts
        'ICC': []  # interlaced (oversampled) cross-correlation images
    }

    drz_sci_fname, drz_sci_ext = parse_file_name(resample.output_sci)
    with fits.open(drz_sci_fname) as hdulist:
        drz_sci = hdulist[drz_sci_ext].data
        drz_wcs = HSTWCS(hdulist, ext=drz_sci_ext)
        if 'EXPTIME' in hdulist[drz_sci_ext].header:
            drz_exptime = hdulist[drz_sci_ext].header['EXPTIME']
        else:
            drz_exptime = hdulist[0].header['EXPTIME']
        drz_units = hdulist[drz_sci_ext].header['BUNIT']
        drz_units = 'rate' if '/' in drz_units else 'counts'

    # get image data, info and create cutouts:
    with fits.open(image) as hdulist:
        for ext in image_ext:
            img_sci = hdulist[ext].data
            img_wcs = HSTWCS(hdulist, ext=ext)
            orig_img_wcs = img_wcs.deepcopy()
            img_exptime = hdulist[0].header['EXPTIME']
            img_units = hdulist[ext].header['BUNIT']
            img_units = 'rate' if '/' in img_units else 'counts'

            if image_sky is not None:
                img_sci -= image_sky[ext]

            imgct_ext, drzct_ext = cutout.create_cutouts(
                primary_cutouts, seg,
                drz_sci, drz_wcs,
                img_sci, img_wcs,
                drz_data_units=drz_units, drz_exptime=drz_exptime,
                flt_data_units=img_units, flt_exptime=img_exptime
            )

            img_info['wcs_info'].append([ext, orig_img_wcs, img_wcs])
            img_info['fits_ext'].extend(len(imgct_ext) * [ext])
            img_info['image_cutouts'].extend(imgct_ext)
            img_info['driz_cutouts'].extend(drzct_ext)

    # find linear fit:
    fit, interlaced_cc, nonshifted_blts = find_linear_fit(
        img_cutouts=imgct_ext,
        drz_cutouts=drzct_ext,
        wcslin=wcslin,
        fitgeom=fitgeom,
        nclip=nclip,
        sigma=sigma
    )

    img_info['blotted_cutouts'].extend(nonshifted_blts)
    img_info['ICC'].extend(interlaced_cc)

    print("\nComputed '{:s}' fit for image {:s}:".format(fitgeom, image))

    if fitgeom == 'shift':
        print("XSH: {:.4f}  YSH: {:.4f}"
              .format(fit['offset'][0], fit['offset'][1]))

    elif fitgeom == 'rscale' and fit['proper']:
        print("XSH: {:.4f}  YSH: {:.4f}    ROT: {:.10g}    SCALE: {:.6f}"
              .format(fit['offset'][0], fit['offset'][1],
                      fit['rot'], fit['scale'][0]))

    elif (fitgeom == 'general'
          or (fitgeom == 'rscale' and not fit['proper'])):
        print("XSH: {:.4f}  YSH: {:.4f}    PROPER ROT: {:.10g}    "
              .format(fit['offset'][0], fit['offset'][1], fit['rot']))

        print("<ROT>: {:.10g}  SKEW: {:.10g}    ROT_X: {:.10g}  "
              "ROT_Y: {:.10g}".format(fit['rotxy'][2], fit['skew'],
                                      fit['rotxy'][0], fit['rotxy'][1]))

        print("<SCALE>: {:.10g}  SCALE_X: {:.10g}  SCALE_Y: {:.10g}"
              .format(fit['scale'][0], fit['scale'][1], fit['scale'][2]))

    print('XRMS: {:.3g}    YRMS: {:.3g}\n'.format(*fit['rms']))
    nmatch = fit['resids'].shape[0]
    print('Final solution based on {:d} objects.'.format(nmatch))

    # correct WCS:
    for ext, owcs, wcs in img_info['wcs_info']:
        correct_wcs(imwcs=wcs, wcslin=drz_wcs, rotmat=fit['fit_matrix'],
                    shifts=fit['offset'], fitgeom=fitgeom)

        print("\n------- ORIGINAL WCS for '{:s}[{}]': ------"
              .format(image, ext))
        print(owcs)

        print("\n------- CORRECTED WCS for '{:s}[{}]': ------"
              .format(image, ext))
        print(wcs)

    return fit, img_info


def find_linear_fit(img_cutouts, drz_cutouts, wcslin=None, fitgeom='general',
                    nclip=3, sigma=3.0):
    """
    Perform linear fit to diplacements (found using cross-correlation) between
    ``img_cutouts`` and "blot" of ``drz_cutouts`` onto ``img_cutouts``.

    Parameters
    ----------
    img_cutouts : Cutout
        Cutouts whose WCS should be aligned.

    drz_cutouts : Cutout
        Cutouts that serve as "reference" to which ``img_cutouts`` will be
        aligned.

    wcslin : astropy.wcs.WCS, None, optional
        A `~astropy.wcs.WCS` object that does not have non-linear distortions.
        This WCS defines a tangen plane in which image alignemnt will be
        performed. When not provided or set to `None`,
        internally ``wcslin`` will be set to ``drz_cutouts[0].wcs``.

    fitgeom : {'shift', 'rscale', 'general'}, optional
        The fitting geometry to be used in fitting cutout displacements.
        This parameter is used in fitting the offsets, rotations
        and/or scale changes from the matched object lists. The 'general'
        fit geometry allows for independent scale and rotation for
        each axis.

    nclip : int, optional
        Number (a non-negative integer) of clipping iterations in fit.

    sigma : float, optional
        Clipping limit in sigma units.

    Returns
    -------
    fit : dict
        A dictionary of various fit parameters computed during the fit. Use
        ``fit.keys()`` to find which parameters are being returned.

    interlaced_cc : numpy.ndarray
        Interlaced (super-sampled) cross-correlation image. This is provided
        as a diagnostic tool for debugging purposes.

    nonshifted_blts : Cutout
        A list of cutouts of blotted ``drz_cutouts`` without applying any
        sub-pixel shifts. This is provided as a diagnostic tool for
        debugging purposes.

    """
    # check that number of drizzled and FLT cutouts match:
    if not isinstance(img_cutouts, collections.Iterable):
        img_cutouts = [img_cutouts]

    if not isinstance(drz_cutouts, collections.Iterable):
        drz_cutouts = [drz_cutouts]

    if len(img_cutouts) != len(drz_cutouts):
        raise ValueError("The number of image cutouts must match the number "
                         "of drizzled cutouts.")

    # choose a tangent plane WCS if not provided:
    if wcslin is None:
        # get wcs from the first drzizzled cutout:
        wcslin = drz_cutouts[0].wcs
    wcslin = wcslin.deepcopy()

    # check if reference WCS is distorted:
    if any(getattr(wcslin, distortion) is not None
           for distortion in _ASTROPY_WCS_DISTORTIONS):
        raise ValueError("Reference WCS must not have non-linear distortions.")

    xyref = np.empty((len(img_cutouts), 2), dtype=np.float)
    xy = np.empty_like(xyref)
    img_dxy = np.empty_like(xyref)
    ref_dxy = np.empty_like(xyref)

    interlaced_cc = []
    nonshifted_blts = []

    # create shifted FLT cutouts:
    for k, (imct, dzct) in enumerate(zip(img_cutouts, drz_cutouts)):
        # store intitial image cutout displacements:
        dx0 = imct.dx
        dy0 = imct.dy

        dzct.data[dzct.mask] = 0

        # blot to initial image cutout:
        blt00 = blot_cutout(dzct, imct)

        # blot to a shifted image cutout by 1/2 along X:
        imct.dx -= 0.5
        blt10 = blot_cutout(dzct, imct)

        # blot to a shifted image cutout by 1/2 along X and Y:
        imct.dy -= 0.5
        blt11 = blot_cutout(dzct, imct)

        # blot to a shifted image cutout by 1/2 along Y:
        imct.dx = dx0
        blt01 = blot_cutout(dzct, imct)

        # restore original image cutout's displacement:
        imct.dy = dy0

        # find cross-correlation shift in image's coordinate system:
        dx, dy, icc, _ = cc.find_displacement(
            imct.data, blt00.data, blt10.data, blt01.data, blt11.data,
            full_output=True
        )

        interlaced_cc.append(icc)
        nonshifted_blts.append(blt00)
        img_dxy[k] = [-dx, -dy]

        # convert displacements to reference linear WCS's image coordinates:
        x1, y1 = imct.cutout_src_pos
        xyref[k] = np.array(wcslin.wcs_world2pix(*imct.pix2world(x1, y1), 0))

        x2 = x1 - dx
        y2 = y1 - dy
        xy[k] = np.array(wcslin.wcs_world2pix(*imct.pix2world(x2, y2), 0))

        ref_dxy[k] = xy[k] - xyref[k]

    # find linear transformation:
    fit = linearfit.iter_fit_all(
        xy, xyref, xyindx=np.arange(len(xy)), uvindx=np.arange(len(xy)),
        mode=fitgeom, center=np.array(wcslin.wcs.crpix),
        nclip=nclip, sigma=sigma, verbose=False
    )

    #TODO: possibly compute fit RMS in the *image* coordinate system
    #      (using input image's pixel scale).

    fit['subpixal_img_dxy'] = img_dxy
    fit['subpixal_ref_dxy'] = ref_dxy

    return fit, interlaced_cc, nonshifted_blts


def correct_wcs(imwcs, wcslin, rotmat, shifts, fitgeom):
    """ Correct input WCS using supplied linear transformations defined in a
    linear WCS. This function modifies ``imwcs`` with the corrected WCS
    parameters.

    """
    updatehdr.update_refchip_with_shift(
        imwcs, wcslin, fitgeom=fitgeom, xsh=shifts[0], ysh=shifts[1],
        fit=rotmat
    )


def update_image_wcs(image, ext, wcs, wcsname=None):
    """
    Updates the WCS of the specified extension with the new WCS
    after archiving the original WCS.

    Parameters
    ----------
    image : str, astropy.io.fits.HDUList
        Filename of image with WCS that needs to be updated

    ext : int, str or tuple of (string, int)
       The key identifying the HDU.  If ``ext`` is a tuple, it is of the
       form ``(name, ver)`` where ``ver`` is an ``EXTVER`` value that must
       match the HDU being searched for.

       If the key is ambiguous (e.g. there are multiple 'SCI' extensions)
       the first match is returned.  For a more precise match use the
       ``(name, ver)`` pair.

       If even the ``(name, ver)`` pair is ambiguous (it shouldn't be
       but it's not impossible) the numeric index must be used to index
       the duplicate HDU.

    wcs : object
        Full HSTWCS object which will replace/update the existing WCS

    wcsname : str, None, optional
        Label to give newly updated WCS. The default value will set the
        WCS name to `SUBPIXAL`.

    """
    close = False

    try:
        if not isinstance(image, fits.HDUList):
            image = fits.open(image, mode='update')
            close = True

        extnum = image.index_of(ext)

        if wcsname is None or wcsname.strip().upper() in ['', 'NONE', 'INDEF']:
            wcsname = 'SUBPIXAL'

        updatehdr.update_wcs(image, extnum, wcs, wcsname=wcsname,
                             reusename=True, verbose=False)

    except:
        raise

    finally:
        if close:
            image.close()
