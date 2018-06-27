"""
A module that manages catalogs and source finding algorithms (i.e.,
``SExtractor`` source finding).

:Author: Mihai Cara (contact: help@stsci.edu)

:License: :doc:`../LICENSE`

"""
from __future__ import (absolute_import, division, unicode_literals,
                        print_function)

import copy
import sys
import subprocess

import six
import numpy as np
from astropy.io import ascii as ascii_io

from stsci.skypac import parseat


__all__ = ['SourceCatalog', 'SExCatalog', 'SExImageCatalog']


_INT_TYPE = (int, long,) if sys.version_info < (3,) else (int,)


def _is_int(n):
    return (
        (isinstance(n, _INT_TYPE) and not isinstance(n, bool)) or
        (isinstance(n, np.generic) and np.issubdtype(n, np.integer))
    )


def _py2round(x):
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


class SourceCatalog(object):
    """
    A class for handling catalog data: storing, filtering, and retrieving
    sources.


    """

    _CMP_MAP = {
        '>': np.greater,
        '>=': np.greater_equal,
        '==': np.equal,
        '!=': np.not_equal,
        '<': np.less,
        '<=': np.less_equal
    }

    # NOTE: Keep all predefined catalog keys ("catalog type", e.g.,
    # 'SExtractor') in _PREDEF_CATMAP in LOWER CASE!
    _PREDEF_CATMAP = {
        'sextractor': {
            'NUMBER':'id',
            'X_IMAGE': 'x',
            'Y_IMAGE': 'y',
            'FLUX_PETRO': 'flux',
            'PETRO_RADIUS': 'radius',
            'A_IMAGE': 'prof-rms-a',
            'B_IMAGE': 'prof-rms-b',
            'FWHM_IMAGE': 'fwhm',
            'CLASS_STAR': 'stellarity'
        }
    }

    def __init__(self):
        self._rawcat = None
        self._rawcat_colnames = None
        self._origin = 0
        self._catalog = None
        self._mask = None
        self._dirty = False
        self._required_colnames = [
            'id', 'x', 'y', 'flux', 'radius', 'prof-rms-a', 'prof-rms-b',
            'stellarity', 'fwhm'
        ]
        self._catmap = {i: i for i in self._required_colnames}
        self.set_default_filters()

    @property
    def predefined_catmaps(self):
        """ Get names of available (pre-defined) column name mappings. """
        return list(self._PREDEF_CATMAP.keys())

    def set_default_filters(self):
        """ Set default source selection criteria. """
        self._filters = [
            ('flux', '>', 0), ('fwhm', '>', 0), ('radius', '>', 0),
            ('prof-rms-a', '>', 0), ('prof-rms-b', '>', 0)
        ]

    def mark_dirty(self):
        """ Mark the catalog as "dirty", indicating whether or not sources
        should be re-extracted from the raw catalog when ``execute()`` is
        run. Masking and filtering criteria will be applied to the raw catalog
        during this run of ``execute()``.
        """
        self._dirty = True

    def is_dirty(self):
        """ Returns the "dirty" status.
        When a catalog is marked as "dirty", sources must be (re-)extracted
        from the raw catalog. In order to update

        """
        return self._dirty

    def set_raw_catalog(self, rawcat, catmap=None, origin=0):
        """
        Parameters
        ----------
        rawcat : astropy.table.Table
            An :py:class:`~astropy.table.Table` containing source data.

        catmap : dict, str, optional
            A `dict` that provides mapping (a dictionary) between source's
            ``'x'``, ``'y'``, ``'flux'``, etc. and the corresponding
            column names in a :py:class:`~astropy.table.Table` catalog.
            Instead of a dictionary, this parameter may be a pre-defined
            mapping name supported by this class. To get the list of
            all pre-defined catalog mapping supported by this class, use
            `SourceCatalog`'s `predefined_catmaps` property. When ``catmap``
            is `None`, no column name mapping will be applied.

        origin: int, float, optional
            Coordinates of the sources of `SourceCatalog.catalog` are
            zero-based. The ``origin`` is used for converting ``rawcat``'s
            coordinates when raw catalog's source coordinates are not
            zero-based.

        """
        if rawcat is None:
            self._rawcat = None
            self._rawcat_colnames = None
            self._catalog = None
            self._mask = None
            self._mask_type = None
            self.set_default_filters()
            self._dirty = False
            return

        if catmap is None:
            catmap = {}
        elif isinstance(catmap, six.string_types):
            catmap = self._PREDEF_CATMAP[catmap.lower()]

        # make sure we have all required columns before we proceed any further:
        new_col_names = [catmap.get(k, k) for k in rawcat.colnames]
        if not all(i in new_col_names for i in self._required_colnames):
            raise ValueError("Input raw catalog does not include all the "
                             "required columns.")

        self._rawcat = rawcat.copy()
        self._dirty = True

        self._origin = origin
        self._catmap = {}

        if catmap is not None:
            colnames = rawcat.colnames
            self._catmap.update(catmap)
            for k, v in catmap.items():
                if k in colnames:
                    self._rawcat.rename_column(k, v)

        # reset mask and filters:
        self._rawcat_colnames = new_col_names
        self.mask = None
        self.set_default_filters()

    @property
    def rawcat(self):
        """ Get raw catalog. """
        return self._rawcat.copy()

    @property
    def required_colnames(self):
        """ Get a list of the minimum column names that are *required* to be
        present in the raw catalog **after** catalog column name mapping has
        been applied.

        """
        return self._required_colnames[:]

    @property
    def rawcat_colnames(self):
        """ Get a list of the column names in the raw catalog **after**
        catalog column name mapping has been applied.

        """
        return self._rawcat_colnames[:]

    @property
    def catmap(self):
        """ Get raw catalog column name mapping. """
        return {k: v for k, v in self._catmap}

    @property
    def mask_type(self):
        """ Get mask type: 'coords', 'image', or `None` (mask not set). """
        return self._mask_type

    @property
    def mask(self):
        """ Get mask indicating "bad" (`True`) and "good" (`False`) sources
        when ``mask_type`` is ``'image'`` or a 2D array of shape ``(N, 2)``
        containing integer coordinates of "bad" pixels.

        """
        return None if self._mask is None else self._mask.copy()

    @mask.setter
    def mask(self, mask):
        """
        Get/Set mask used to ignore (mask) "bad" sources from the raw catalog.
        Mask is a 2D image-like (but boolean) `numpy.ndarray` indicating
        "bad" pixels using value `True` (=ignore these pixels) and  "good"
        pixels using the value `False` (=no need to mask).

        Parameters
        ----------

        mask : str, tuple of two 1D lists of int, 2D numpy.ndarray
            A mask can be provided in several ways:

            - When ``mask`` is a string, it is assumed to be the name of
              a simple FITS file contaning a boolean mask indicating
              "bad" pixels using value `True` and  "good" pixels using
              value `False` (=no need to mask).

            - ``mask`` can also be provided directly as a boolean 2D "image"
              in the form of a boolean `numpy.ndarray`.

            - Finally, ``mask`` can be a tuple of exactly two lists (or 1D
              `numpy.ndarray`) containing **integer** coordinates of the
              "pixels" to be masked as "bad". Any source with coordinates
              within such a "pixel" will be excluded from the catalog.

        """
        if mask is None:
            if self._mask is not None:
                self._dirty = True
            self._mask = None
            self._mask_type = None
            return

        elif isinstance(mask, six.string_types):
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

        self._dirty = True

    def set_filters(self, fcond):
        """
        Set conditions for *selecting* sources from the raw catalog.

        Parameters
        ----------

        fcond : tuple, list of tuples
            Each selection condition must be specified as a tuple of the form
            ``(colname, comp, value)`` where:

            - ``colname`` is a column name from the raw catalog **after**
              catalog column name mapping has been applied. Use
              `rawcat_colnames` to get a list of available column names.

            - ``comp`` is a **string** representing a comparison operator.
              The following operators are suported:
              ``['>', '>=', '==', '!=', '<', '<=']``.

            - ``value`` is a numeric value to be used for comparison of column
              values.

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
            self._dirty = True
            self._filters = filters

    def reset_filters(self):
        """ Remove selection filters. """
        self._dirty = len(self._filters) > 0
        self._filters = []

    def append_filters(self, fcond):
        """
        Add one or more conditions for *selecting* sources from the raw
        catalog to already set filters. See :py:meth:`set_filters` for
        description of parameter ``fcond``.

        """
        self._dirty = True

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
            self._dirty = True

        else:
            raise TypeError("'fcond' must be a tuple or a list of tuples.")

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

        self._dirty = True

    @property
    def filters(self):
        """ Get a list of all active selection filters. """
        return self._filters[:]

    @classmethod
    def _op2cmp(cls, op):
        op = ''.join(op.split())
        return cls._CMP_MAP[op]

    def execute(self):
        """ Compute catalog applying masks and selecting only sources
        that satisfy all set filters.
        """
        if not self._dirty:
            return

        # start with the original "raw" catalog:
        catalog = self._rawcat.copy()

        # remove sources with masked values:
        if catalog.masked:
            mask = np.zeros(len(catalog), dtype=np.bool)
            for c in catalog.itercols:
                mask = np.logical_or(mask, c.mask)

            # create a new catalog having only the "good" data without mask:
            catalog = catalog.__class__(catalog[np.logical_not(mask)],
                                        masked=False)

        # correct for 'origin':
        catalog['x'] -= self._origin
        catalog['y'] -= self._origin
        xi = _py2round(np.asarray(catalog['x'])).astype(np.int)
        yi = _py2round(np.asarray(catalog['y'])).astype(np.int)

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
            if key in catalog.colnames:
                mask *= self._op2cmp(op)(catalog[key], val)

        catalog = catalog[mask]
        xi = xi[mask]
        yi = yi[mask]
        calc = np.asarray(catalog['flux'])**2 / np.asarray(catalog['fwhm'])
        Rup = _py2round(np.asarray(catalog['radius'])).astype(np.int)
        Aup = _py2round(np.asarray(catalog['prof-rms-a'])).astype(np.int)
        Rlg = 2 * Rup * Aup
        xs = xi - Rlg
        xe = xi + Rlg
        ys = yi - Rlg
        ye = yi + Rlg

        catalog['xi'] = xi
        catalog['yi'] = yi
        catalog['xs'] = xs
        catalog['xe'] = xe
        catalog['ys'] = ys
        catalog['ye'] = ye
        catalog['calc'] = calc

        self._catalog = catalog
        self._dirty = False

    @property
    def catalog(self):
        """ Get catalog (after applying masks and selection filters). """
        if self._dirty:
            self.execute()
        return self._catalog


class SExCatalog(SourceCatalog):
    """ A catalog class specialized for handling ``SExtractor`` output
    catalogs, such as being able to load raw ``SExtractor`` catalogs
    directly from text files.

    Parameters
    ----------
    rawcat : astropy.table.Table, str
        An :py:class:`~astropy.table.Table` containing source data or a
        ``SExtractor``-generated text file name.

    max_stellarity : float, optional
        Maximum stellarity for selecting sources from the catalog.

    """
    def __init__(self, rawcat=None, max_stellarity=1.0):
        self._max_stellarity = max_stellarity
        super().__init__()
        self._origin = 1
        self.set_raw_catalog(rawcat)

    def set_default_filters(self):
        """ Sets default filters for selecting sources from the raw catalog.

        Default selection criteria are: ``flux > 0`` AND ``fwhm > 0`` AND
        ``radius > 0`` AND ``prof-rms-a > 0`` AND ``prof-rms-b > 0`` AND
        ``stellarity <= max_stellarity``.

        """

        self._filters = [
            ('flux', '>', 0), ('fwhm', '>', 0), ('radius', '>', 0),
            ('prof-rms-a', '>', 0), ('prof-rms-b', '>', 0),
            ('stellarity', '<=', self._max_stellarity)
        ]

    def set_raw_catalog(self, rawcat):
        """ Set/replace raw catalog.

        Parameters
        ----------
        rawcat : astropy.table.Table, str
            An :py:class:`~astropy.table.Table` containing source data or a
            ``SExtractor``-generated text file name.

        """
        if isinstance(rawcat, six.string_types):
            rawcat = ascii_io.read(rawcat, guess=False, format='sextractor')

        super().set_raw_catalog(rawcat, catmap='sextractor', origin=1)


class SExImageCatalog(SExCatalog):
    """ A catalog class specialized for finding sources in images using
    ``SExtractor`` and then loading raw ``SExtractor`` catalogs
    directly from output text files.

    Parameters
    ----------
    image : str
        A ``FITS`` image file name.

    sexconfig : str
        File name of the ``SExtractor`` configuration file to be used for
        finding sources in the ``image``.

    max_stellarity : float, optional
        Maximum stellarity for selecting sources from the catalog.

    sextractor_cmd : str, optional
        Command to invoke ``SExtractor``.

    """
    def __init__(self, image=None, sexconfig=None, max_stellarity=1.0,
                 sextractor_cmd='sex'):
        self._max_stellarity = max_stellarity
        self._sextractor_cmd = sextractor_cmd

        super().__init__(rawcat=None, max_stellarity=max_stellarity)

        self._catname = None
        self.sexconfig = sexconfig
        self.image = image
        # _dirty_img indicates that either image or SExtractor configuration
        # file (or both) have changed and that a re-extraction of sources
        # is needed.
        self._dirty_img = not (image is None or sexconfig is None)

    def _find_sources(self):
        # returns exit status of the child (i.e., 'sex') process
        # returns None if either image or sexconfig are not set:
        if self._image is None or self._sexconfig is None:
            return None

        cmd = [self._sextractor_cmd, self.image, '-c', self._sexconfig]
        # popen = subprocess.Popen(cmd, stdout=subprocess.PIPE,
        #                          stderr=subprocess.PIPE)
        # out, err = popen.communicate()
        return subprocess.call(cmd)

    @property
    def image(self):
        """ Get image. """
        return self._image

    @image.setter
    def image(self, image):
        """ Set/Get image file name. """
        if image is None:
            super().set_raw_catalog(rawcat=None)
            self._dirty_img = False
            return

        self._image = image
        self._dirty_img = self.sexconfig is not None

    @property
    def sexconfig(self):
        """ Get ``SExtractor`` configuration file. """
        return self._sexconfig

    @sexconfig.setter
    def sexconfig(self, sexconfig):
        """ Set/Get ``SExtractor`` configuration file. """
        self._sexconfig = sexconfig
        if sexconfig is None:
            super().set_raw_catalog(rawcat=None)
            self._catname = None
            self._dirty_img = False
            self._dirty = False
        else:
            # find output catalog file name:
            self._catname = SExImageCatalog._get_catname(sexconfig)
            self.set_raw_catalog(self._catname)
            self._dirty_img = True
            self._dirty = True

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

        try:
            idx = check_types.index(check_image_type)
        except ValueError:
            return None

        return check_files[idx]

    @property
    def segmentation_file(self):
        """ Get segmentation file name stored in the ``SExtractor``'s
        configuration file or `None`.

        """
        return SExImageCatalog._get_segname(
            self._sexconfig,
            check_image_type='SEGMENTATION'
        )

    def execute(self):
        if self._dirty_img:
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

            self._dirty_img = False

        super().execute()
