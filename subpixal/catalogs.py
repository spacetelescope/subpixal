# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A module that manages catalogs and source finding algorithms (i.e.,
``SExtractor`` source finding).

:Author: Mihai Cara (for help, contact `HST Help Desk <https://hsthelp.stsci.edu>`_)

:License: :doc:`LICENSE`

"""
import os
import sys
import copy
import subprocess
import abc
import tempfile

import numpy as np
from astropy.io import ascii as ascii_io
from astropy.io import fits
from astropy.table import Table

from stsci.skypac import parseat

from . utils import py2round, _create_tmp_fits_file


__all__ = ['ImageCatalog', 'SExImageCatalog']


_FWHM2SIGMA = 2 * np.sqrt(2.0 * np.log(2.0))

def _is_int(n):
    return (
        (isinstance(n, int) and not isinstance(n, bool)) or
        (isinstance(n, np.generic) and np.issubdtype(n, np.integer))
    )


def _select_high(values, nrows):
    idx = [i for i, _ in sorted(enumerate(values), key=lambda v: v[1])]
    idx = idx[-nrows:] if nrows >= 0 else idx[:-nrows]
    mask = np.zeros(len(values), dtype=np.bool_)
    mask[idx] = True
    return mask


def _select_lower(values, nrows):
    idx = [i for i, _ in sorted(enumerate(values), key=lambda v: v[1])]
    idx = idx[:nrows] if nrows >= 0 else idx[nrows:]
    mask = np.zeros(len(values), dtype=np.bool_)
    mask[idx] = True
    return mask


class ImageCatalog(abc.ABC):
    """
    A class for finding sources in images and handling catalog data: storing,
    filtering, and retrieving sources.

    """
    _CMP_MAP = {
        '>': np.greater,
        '>=': np.greater_equal,
        '==': np.equal,
        '!=': np.not_equal,
        '<': np.less,
        '<=': np.less_equal,
        'h': _select_high,
        'l': _select_lower,
    }

    # NOTE: Keep all predefined catalog keys ("catalog type", e.g.,
    # 'SExtractor') in _PREDEF_CATMAP in LOWER CASE!
    _PREDEF_CATMAP = {
        'sextractor': {
            'NUMBER':'id',
            'X_IMAGE': 'x',
            'Y_IMAGE': 'y',
            'FLUX_ISO': 'flux',
            'FLUXERR_ISO': 'fluxerr',
            'A_IMAGE': 'semi-major-a',
            'B_IMAGE': 'semi-major-b',
            'FWHM_IMAGE': 'fwhm',
            'CLASS_STAR': 'stellarity'
        }
    }

    def __init__(self):
        self._filters = []
        self._required_colnames = [
            'id', 'x', 'y', 'flux', 'fluxerr', 'semi-major-a', 'semi-major-b',
            'stellarity', 'fwhm'
        ]
        self._derived_colnames = ['pos_std', 'weight']
        self.set_default_filters()

        self._image = None
        self._image_ext = None
        colnames = self._required_colnames + self._derived_colnames
        self._catalog = self._empty_catalog()

    def _empty_catalog(self):
        colnames = self._required_colnames + self._derived_colnames
        cat = Table(
            names=colnames,
            dtype=[np.int64] + (len(colnames) - 1) * [np.float64]
        )
        return cat

    def set_image(self, image):
        """
        Set image to be used for source finding.

        Parameters
        ----------
        image: numpy.ndarray, str
            When setting an image either a `numpy.ndarray` of image data or
            a string file name is acceptable. Image file name may be followed
            by an extension specification such as ``'file1.fits[1]'`` or
            ``'file1.fits[(sci,1)]'`` (by default, the first image-like
            extension will be used).

        """
        self._image = image
        self._image_ext = None

        if isinstance(image, str):
            files = parseat.parse_cs_line(
                image, default_ext='*', clobber=False, fnamesOnly=False,
                doNotOpenDQ=True, im_fmode='readonly', dq_fmode='readonly',
                msk_fmode='readonly', logfile=None, verbose=False
            )

            if len(files) > 1:
                for f in files:
                    f.release_all_images()
                raise ValueError("Only a single file can be specified as "
                                 "an image.")

            # get extension number
            self._image_ext = files[0].image.hdu.index_of(files[0].fext[0])
            files[0].release_all_images()

    @property
    def image_extn(self):
        """ Get image extension number when the image was set using a string
        file name. When image was set (in py:meth:`set_image`) using a
        `numpy.ndarray`, this property is `None`.

        """
        return self._image_ext

    def set_default_filters(self):
        """ Set default source selection criteria. """
        self._filters = [
            ('flux', '>', 0), ('fwhm', '>', 0),
            ('semi-major-a', '>', 0), ('semi-major-b', '>', 0)
        ]

    @property
    def required_colnames(self):
        """ Get a list of the minimum column names that are *required* to be
        present in the raw catalog **after** catalog column name mapping has
        been applied.

        """
        return self._required_colnames[:]

    @property
    def mask_type(self):
        """ Get mask type: 'coords', 'image', or `None` (mask not set). """
        return self._mask_type

    def set_mask(self, mask):
        """ Get/Set mask used to ignore (mask) "bad" sources from the catalog.

        Parameters
        ----------

        mask : str, tuple of two 1D lists of int, 2D numpy.ndarray
            A mask can be provided in several ways:

            - When ``mask`` is a string, it is assumed to be the name of
              a simple FITS file contaning a boolean mask indicating
              "bad" pixels using value `True` (=ignore these pixels) and
              "good" pixels using value `False` (=no need to mask).

            - ``mask`` can also be provided directly as a boolean 2D "image"
              in the form of a boolean `numpy.ndarray`.

            - Finally, ``mask`` can be a tuple of exactly two lists (or 1D
              `numpy.ndarray`) containing **integer** coordinates of the
              "pixels" to be masked as "bad". Any source with coordinates
              within such a "pixel" will be excluded from the catalog.

        """
        if mask is None:
            self._mask = None
            self._mask_type = None
            return

        elif isinstance(mask, str):
            # open file and find image extensions:
            files = parse_cs_line(mask, default_ext='*', clobber=False,
                                  fnamesOnly=False, doNotOpenDQ=True,
                                  im_fmode='readonly', dq_fmode='readonly',
                                  msk_fmode='readonly', logfile=None,
                                  verbose=False)

            if len(files) > 1:
                for f in files:
                    f.release_all_images()
                raise ValueError("Only a single file can be specified as mask")

            self._mask = np.array(files[0].image.hdu[files[0].fext].data,
                                  dtype=np.bool)
            self._mask_type = 'image'
            files[0].release_all_images()

        elif isinstance(mask, tuple):
            if len(mask) != 2:
                raise ValueError("When 'mask' is a tuple, it must contain "
                                 "two 1D lists of integer coordinates to be "
                                 "excluded from the catalog.")

            x, y = mask
            x = np.asarray(x)
            y = np.asarray(y)

            if len(x.shape) != 1 or x.shape != y.shape or not _is_int(x[0]) \
               or not _is_int(y[0]):
                raise ValueError("When 'mask' is a tuple, it must contain "
                                 "two 1D lists of equal length of integer "
                                 "coordinates to be excluded from the "
                                 "catalog.")
            self._mask = np.array([x, y]).T
            self._mask_type = 'coords'

        else:
            mask = np.array(mask)
            if len(mask.shape) == 2 and mask.shape[1] == 2 and \
               np.issubdtype(mask.dtype, np.integer):
                # we are dealing with a "list" of integer indices:
                self._mask = mask
                self._mask_type = 'coords'

                #nonneg = np.prod(mask >= 0, axis=1, dtype=np.bool)
                #mask = mask[nonneg]
                #badpix = tuple(np.fliplr(mask.T))
                #self._mask = np.ones(np.max(mask, axis=0) + 1, dtype=np.bool)
                #self._mask[badpix] = False

            elif len(mask.shape) == 2 and np.issubdtype(mask.dtype, np.bool):
                # we are dealing with a boolean mask:
                self._mask = mask
                self._mask_type = 'image'

            else:
                raise ValueError("Unsupported mask type or format.")

    def set_filters(self, fcond):
        """
        Set conditions for *selecting* sources from the raw catalog.

        Parameters
        ----------

        fcond : tuple, list of tuples
            Each selection condition must be specified as a tuple of the form
            ``(colname, cond, value)`` OR ``(colname, nrows)`` where:

            - ``colname`` is a column name from the raw catalog **after**
              catalog column name mapping has been applied. Use
              `rawcat_colnames` to get a list of available column names.

            - ``cond`` is a **string** representing a selection condition,
              i.e., a comparison operator. The following operators are
              suported: ``['>', '>=', '==', '!=', '<', '<=', 'h', 'l']``. The
              ``'h'`` or ``'l'`` operators are used to select a specific
              number of rows (specified by the ``value``) that have highest
              or lowest values in the column specified by ``colname``.
              Selection of highest/lowest values is performed last, after all
              other comparison-based filters have been applied.

            - ``value`` is a numeric value to be used for comparison of column
              values. When ``cond`` is either ``'h'`` or ``'l'``, this value
              must be a *positive integer* number of rows to be .

            Multiple selection conditions can be provided as a list of the
            condition tuples described above.

        """
        if isinstance(fcond, list):
            filters = []

            for f in fcond[::-1]:
                key, op, val = f[:3]
                op = ''.join(op.split())
                idxs = self._find_filters(filters, key, op)
                if idxs is None:
                    filters.insert(0, (key, op, val))

        elif isinstance(fcond, tuple):
            key, op, val = fcond[:3]
            op = ''.join(op.split())
            filters = [(key, op, val)]

        else:
            raise TypeError("'fcond' must be a tuple or a list of tuples.")

        if self._filters != filters:
            self._filters = filters

    def append_filters(self, fcond):
        """
        Add one or more conditions for *selecting* sources from the raw
        catalog to already set filters. See :py:meth:`set_filters` for
        description of parameter ``fcond``.

        """
        if isinstance(fcond, list):
            for f in fcond:
                key, op, val = f[:3]
                op = ''.join(op.split())
                flt = (key, op, val)
                idxs = self._find_filters(self._filters, key, op)
                if idxs is not None:
                    for i in idxs:
                        del self._filters[i]
                self._filters.append((key, op, val))

        elif isinstance(fcond, tuple):
            key, op, val = fcond[:3]
            op = ''.join(op.split())
            idxs = self._find_filters(self._filters, key, op)
            if idxs is not None:
                for i in idxs:
                    del self._filters[i]
            self._filters.append((key, op, val))

        else:
            raise TypeError("'fcond' must be a tuple or a list of tuples.")

    def remove_all_filters(self):
        """ Remove all selection filters. """
        self._filters = []

    @staticmethod
    def _find_filters(filters, key, op=None):
        idxs = []
        for i, (k, o, _) in enumerate(filters):
            if k == key and (op is None or o == op):
                idxs.append(i)
        return idxs if idxs else None

    def remove_filter(self, key, op=None):
        """ Remove a specific filter by column name and, optionally, by
        comparison operator.

        Parameters
        ----------
        key : str
            Column name to which selection criteria (filter) is applied.
            If more conditions match a column name vale, all of them will be
            removed.

        op : str, optional
            Specifies the comparison operation used in a filter. This allows
            narrowing down which filters should be removed.

        """
        idxs = self._find_filters(self._filters, key, op)
        if idxs is None:
            return

        for idx in idxs[::-1]:
            del self._filters[idx]

    @property
    def filters(self):
        """ Get a list of all active selection filters. """
        return self._filters[:]

    @classmethod
    def _op2cmp(cls, op):
        op = ''.join(op.split())
        return cls._CMP_MAP[op]

    @abc.abstractmethod
    def execute(self):
        """ Find sources in the image. Compute catalog applying masks and
        selecting only sources that satisfy all set filters.
        """
        pass

    def catalog(self):
        """ Get catalog (after applying masks and selection filters). """
        return self._catalog

    def get_segmentation_image(self):
        """ Get segmentation image used to identify catalog's sources. """
        return None

    def compute_position_std(self, catalog):
        """ This function is called to compute source position error estimate.
        This function uses the following simplified estimate:
        :math:`\sigma_{\mathrm{pos}}=\sigma_{\mathrm{Gaussian}} / \mathrm{SNR}=\mathrm{FWHM}/(2\sqrt{2 \ln 2}\mathrm{SNR})`.
        Sub-classes can implement more accurate position error computation.

        Parameters
        ----------
        catalog : astropy.table.Table
            A table containing `~ImageCatalog.required_colnames` columns.

        Returns
        -------
        pos_std : numpy.ndarray
            Position error computed from input catalog data.

        """
        if self._catalog is None:
            return None
        pos_std = np.asarray(
            catalog['fwhm'] / (_FWHM2SIGMA * catalog['flux'] /
                               catalog['fluxerr'])
        )
        return pos_std

    def compute_weights(self, catalog):
        """ This function is called to compute source weights in a catalog.
        Currently, all weights are set equal to 1. Sub-classes should implement
        more meaningful weight computation.

        Parameters
        ----------
        catalog : astropy.table.Table
            A table containing `~ImageCatalog.required_colnames` columns.

        Returns
        -------
        weights : numpy.ndarray
            Weights computed from input catalog data.

        """
        if self._catalog is None:
            return None
        weights = np.ones(len(self._catalog), dtype=np.float)
        return weights


class SExImageCatalog(ImageCatalog):
    """ A catalog class specialized in finding sources using ``SExtractor``
    and then loading and processing raw ``SExtractor`` catalogs and its output
    files.

    Parameters
    ----------
    image : str
        A ``FITS`` image file name.

    sexconfig : str
        File name of the ``SExtractor`` configuration file to be used for
        finding sources in the ``image``.

    max_stellarity : float, None, optional
        Maximum stellarity for selecting sources from the catalog. When
        ``max_stellarity`` is `None`, source filtering by 'stellarity' is
        turned off.

    sextractor_cmd : str, optional
        Command to invoke ``SExtractor``.

    """
    def __init__(self, image=None, sexconfig=None, max_stellarity=1.0,
                 sextractor_cmd='sex'):
        # Check SExtractor command:
        if not isinstance(sextractor_cmd, str):
            raise TypeError("SExtractor command must be a string.")
        sextractor_cmd = sextractor_cmd.strip()
        if not sextractor_cmd:
            ValueError("SExtractor command must be a non-empty string.")

        self._max_stellarity = max_stellarity
        super().__init__()

        # self._dirty_image indicates whether SExtractor needs
        # to be re-run on the image.
        self._dirty_image = False

        # self._dirty_filters indicates whether selection criteria
        # need to be re-applied.
        self._dirty_filters = False

        self._sextractor_cmd = sextractor_cmd
        self._sexconfig = sexconfig

        self._mask = None
        self._mask_type = None

        self._file_name = None
        self._tmp_file = None
        self.set_image(image)

        self._reset_catalogs()

    def set_default_filters(self):
        """ Sets default filters for selecting sources from the raw catalog.

        Default selection criteria are: ``flux > 0`` AND ``fwhm > 0`` AND
        ``semi-major-a > 0`` AND ``semi-major-b > 0`` (AND
        ``stellarity <= max_stellarity``, if ``max_stellarity`` is not `None`).

        """
        filters = [
            ('flux', '>', 0), ('fwhm', '>', 0),
            ('semi-major-a', '>', 0), ('semi-major-b', '>', 0)
        ]
        if self._max_stellarity is not None:
            filters.append(('stellarity', '<=', self._max_stellarity))

        self._dirty_filters = SExImageCatalog._filters_changed(
            filters, self._filters
        )
        self._filters = filters

    @staticmethod
    def _filters_changed(filter1, filter2):
        # TODO: implement real comparison of filter rules
        return True

    def _reset_catalogs(self):
        self._sexcat = None
        self._catalog = self._empty_catalog()
        self._dirty_image = self._image is not None
        self._dirty_filters = (self._dirty_image and
                               self._sexconfig is not None)

    def _set_sex_catalog(self, sexcat):
        """ Set/replace raw catalog as extracted from SExtractor's output file.

        Parameters
        ----------
        sexcat : str
            A ``SExtractor``-generated text file name.

        """
        if isinstance(sexcat, str):
            sexcat = ascii_io.read(sexcat, guess=False, format='sextractor')
        else:
            raise TypeError("Unsupported SExtractor catalog type.")

        catmap = self._PREDEF_CATMAP['sextractor']

        # make sure we have all required columns before we proceed any further:
        new_col_names = [catmap.get(k, k) for k in sexcat.colnames]
        if not all(i in new_col_names for i in self._required_colnames):
            raise ValueError("Input raw catalog does not include all the "
                             "required columns.")

        self._sexcat = sexcat
        self._catalog = self._empty_catalog()
        self._dirty_filters = True

        for k, v in catmap.items():
            if k in sexcat.colnames:
                self._sexcat.rename_column(k, v)

    def _find_sources(self):
        # returns exit status of the child (i.e., 'sex') process
        # returns None if either image or sexconfig are not set:
        if (not self._dirty_image or self._image is None or
            self._sexconfig is None):
            return None

        # delete output files:
        catname = SExImageCatalog._get_catname(self._sexconfig)
        files = SExImageCatalog._get_checkname(self._sexconfig, None)
        files = [catname] if files is None else [catname] + files
        for f in files:
            if os.path.isfile(f):
                os.remove(f)

        cmd = [self._sextractor_cmd, self._file_name, '-c', self._sexconfig]
        # popen = subprocess.Popen(cmd, stdout=subprocess.PIPE,
        #                          stderr=subprocess.PIPE)
        # out, err = popen.communicate()
        return subprocess.call(cmd)

    def set_image(self, image):
        """
        Set image to be used for source finding.

        Parameters
        ----------
        image: numpy.ndarray, str
            When setting an image either a `numpy.ndarray` of image data or
            a string file name is acceptable. Image file name may be followed
            by an extension specification such as ``'file1.fits[1]'`` or
            ``'file1.fits[(sci,1)]'`` (by default, the first image-like
            extension will be used).

        """
        super().set_image(image)
        self._reset_catalogs()
        self._file_name = None

        if self._tmp_file is not None:
            self._tmp_file.close()

        if image is None:
            self._tmp_file = None
            return

        if isinstance(self._image, str):
            self._file_name = self._image
            self._tmp_file = None
            return

        if not isinstance(self._image, np.ndarray):
            raise TypeError("'image' must be either a string file name or "
                            "a numpy.ndarray.")

        # We've got a numpy.ndarray image. We need to create a temporary file
        # to be used with SExtractor.
        self._tmp_file = _create_tmp_fits_file(
            fits.HDUList([fits.PrimaryHDU(image)]),
            prefix='tmp_SExImageCatalog_sci_'
        )
        self._file_name = self._tmp_file.name
        self._image_ext = 0

    def set_mask(self, mask):
        super().set_mask(mask)
        self._dirty_filters = True

    @property
    def sexconfig(self):
        """ Set/Get ``SExtractor`` configuration file. """
        return self._sexconfig

    @sexconfig.setter
    def sexconfig(self, sexconfig):
        self._sexconfig = sexconfig
        self._reset_catalogs()

    @staticmethod
    def _get_catname(sexconfig):
        catname = None

        with open(sexconfig, mode='r') as f:
            line = f.readline()

            while line:
                line = line.strip()

                if not line or line.startswith('#'):
                    line = f.readline()
                    continue

                # retrieve the part before inline comments
                cfgpar = line.split('#')[0]
                cfgpar = cfgpar.strip()

                if len(cfgpar) > 1:
                    parname, parval = cfgpar.split()[:2]
                    if parname.upper().startswith('CATALOG_NAME'):
                        catname = parval.strip()
                        break

                line = f.readline()

        if catname is None:
            raise ValueError(
                "Unable to retrieve SExtractor catalog name from the "
                "provided SExtractor configuration file '{:s}'."
                .format(sexconfig)
            )

        return catname

    @staticmethod
    def _get_checkname(sexconfig, check_image_type):
        """
        When ``check_image_type`` is `None` - return a list of all
        "check image" file names. Otherwise, return a single file name
        corresponding to the requested check image type.

        """
        segname = None
        check_types = None
        check_files = None

        with open(sexconfig, mode='r') as f:
            line = f.readline()

            while line and (check_types is None or check_files is None):
                line = line.strip()

                if not line or line.startswith('#'):
                    line = f.readline()
                    continue

                # retrieve the part before inline comments
                cfgpar = line.split('#')[0]
                cfgpar = cfgpar.strip()

                if len(cfgpar) > 1:
                    parname, parval = cfgpar.split()[:2]

                    if cfgpar.upper().startswith('CHECKIMAGE_TYPE'):
                        check_types = list(map(str.upper, map(
                            str.strip,
                            cfgpar[len('CHECKIMAGE_TYPE'):].split(',')
                        )))

                    elif cfgpar.upper().startswith('CHECKIMAGE_NAME'):
                        check_files = list(map(
                            str.strip,
                            cfgpar[len('CHECKIMAGE_NAME'):].split(',')
                        ))

                line = f.readline()

        if check_types is None or check_files is None:
            return None

        if check_image_type is None:
            return check_files

        try:
            idx = check_types.index(check_image_type)
        except ValueError:
            return None

        return check_files[idx]

    def get_segmentation_image(self):
        """ Get segmentation file name stored in the ``SExtractor``'s
        configuration file or `None`.

        """
        if self._sexconfig is None:
            return None

        self.execute()

        filename = SExImageCatalog._get_checkname(
            self._sexconfig,
            check_image_type='SEGMENTATION'
        )

        if os.path.isfile(filename):
            return fits.getdata(filename, ext=0)
        else:
            return None

    @property
    def catalog(self):
        """ Get catalog (after applying masks and selection filters). """
        self.execute()
        return self._catalog

    def execute(self):
        """ Find sources in the image. Compute catalog applying masks and
        selecting only sources that satisfy all set filters.
        """
        if self._dirty_image:
            if self._file_name is None:
                # Unable to run SExtractor: No image set.
                assert(self._image is None)
                self._dirty_filters = False
                return

            if self._sexconfig is None:
                # No SExtractor config file set
                self._dirty_filters = False
                return

            self._dirty_filters = True

            try:
                retcode = self._find_sources()
                if retcode is None:
                    return

                if retcode < 0:
                    print('SExtractor was terminated by signal {:d}'
                          .format(-retcode))
                    return

                elif retcode > 0:
                    print('SExtractor returned {:d}'.format(retcode))
                    return

            except OSError as e:
                print('SExtractor execution failed: {}'.format(e))
                return

            self._dirty_image = False

            self._set_sex_catalog(
                SExImageCatalog._get_catname(self._sexconfig)
            )

        if not self._dirty_filters:
            return

        if self._sexcat is None:
            # SExtractor catalog has not been set.
            # Unable to execute source filtering.
            return

        # start with the original "raw" catalog:
        catalog = self._sexcat.copy()

        # remove sources with masked values:
        if catalog.masked:
            mask = np.zeros(len(catalog), dtype=np.bool)
            for c in catalog.itercols:
                mask = np.logical_or(mask, c.mask)

            # create a new catalog having only the "good" data without mask:
            catalog = catalog.__class__(catalog[np.logical_not(mask)],
                                        masked=False)

        # correct for 'origin':
        catalog['x'] -= 1
        catalog['y'] -= 1
        xi = py2round(np.asarray(catalog['x'])).astype(np.int)
        yi = py2round(np.asarray(catalog['y'])).astype(np.int)

        # apply mask:
        if self._mask is None:
            mask = np.ones(xi.size, dtype=np.bool)

        elif self._mask_type == 'coords':
            mask = np.logical_not(
                [np.any(np.all(np.equal(self._mask, p), axis=1))
                 for p in np.array([xi, yi]).T]
            )

        elif self._mask_type == 'image':
            ymmax, xmmax = self._mask.shape
            mm = (xi >= 0) & (xi < xmmax) & (yi >= 0) & (yi < ymmax)
            mask = np.array(
                [np.logical_not(self._mask[i, j]) if m else False
                 for m, i, j in zip(mm, yi, xi)]
            )

        else:
            raise ValueError("Unexpected value of '_mask_type'. Contact "
                             "software developer.")
        # apply filters:
        for f in self._filters:
            key, op, val = f[:3]
            if op in ['h', 'l']:
                # apply these last:
                continue
            if key in catalog.colnames:
                mask *= self._op2cmp(op)(catalog[key], val)

        catalog = catalog[mask]
        xi = xi[mask]
        yi = yi[mask]

        # compute "derived" quantities:
        pos_std = self.compute_position_std(catalog)
        if pos_std is not None:
            catalog['pos_std'] = pos_std

        weights = self.compute_weights(catalog)
        if weights is not None:
            catalog['weight'] = weights

        # apply mask:
        mask = mask[mask]

        # apply filters (again) to filter for 'pos_std' and other columns:
        for f in self._filters:
            key, op, val = f[:3]
            if op in ['h', 'l']:
                # apply these last:
                continue
            if key in catalog.colnames:
                mask *= self._op2cmp(op)(catalog[key], val)

        # At last, apply top/bottom selection (if any):
        catalog = catalog[mask]
        mask = np.ones(len(catalog), dtype=np.bool)
        for f in self._filters:
            key, op, val = f[:3]
            if op not in ['h', 'l']:
                # skip this as it was already applied
                continue
            if key in catalog.colnames:
                mask *= self._op2cmp(op)(catalog[key], val)

        self._catalog = catalog[mask]
        self._dirty_filters = False


    def set_filters(self, fcond):
        """
        Set conditions for *selecting* sources from the raw catalog.

        Parameters
        ----------

        fcond : tuple, list of tuples
            Each selection condition must be specified as a tuple of the form
            ``(colname, cond, value)`` OR ``(colname, nrows)`` where:

            - ``colname`` is a column name from the raw catalog **after**
              catalog column name mapping has been applied. Use
              `rawcat_colnames` to get a list of available column names.

            - ``cond`` is a **string** representing a selection condition,
              i.e., a comparison operator. The following operators are
              suported: ``['>', '>=', '==', '!=', '<', '<=', 'h', 'l']``. The
              ``'h'`` or ``'l'`` operators are used to select a specific
              number of rows (specified by the ``value``) that have highest
              or lowest values in the column specified by ``colname``.
              Selection of highest/lowest values is performed last, after all
              other comparison-based filters have been applied.

            - ``value`` is a numeric value to be used for comparison of column
              values. When ``cond`` is either ``'h'`` or ``'l'``, this value
              must be a *positive integer* number of rows to be .

            Multiple selection conditions can be provided as a list of the
            condition tuples described above.

        """
        old_filters = self._filters[:]
        super().set_filters(fcond=fcond)
        self._dirty_filters = SExImageCatalog._filters_changed(
            self._filters, old_filters
        )

    def remove_all_filters(self):
        """ Remove all selection filters. """
        self._dirty_filters = bool(self._filters)
        self._filters = []

    def append_filters(self, fcond):
        """
        Add one or more conditions for *selecting* sources from the raw
        catalog to already set filters. See :py:meth:`set_filters` for
        description of parameter ``fcond``.

        """
        old_filters = self._filters[:]
        super().append_filters(fcond=fcond)
        self._dirty_filters = SExImageCatalog._filters_changed(
            self._filters, old_filters
        )

    def remove_filter(self, key, op=None):
        """
        """
        old_filters = self._filters[:]
        super().remove_filter(key=key, op=op)
        self._dirty_filters = SExImageCatalog._filters_changed(
            self._filters, old_filters
        )

    def compute_weights(self, catalog):
        """ This function is called to compute source weights in a catalog.
        This function estimates weights as :math:`1/\sigma_{\mathrm{pos}}`.

        Parameters
        ----------
        catalog : astropy.table.Table
            A table containing `~ImageCatalog.required_colnames` columns.

        Returns
        -------
        weights : astropy.table.Column
            Weights computed from input catalog data.

        """
        pos_std = self.compute_position_std(catalog)
        if pos_std is None:
            return None
        weights = 1.0 / pos_std**2
        return weights
