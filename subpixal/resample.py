
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A module that manages resampling of images onto a common output frame
and also "inverse" blotting.

:Author: Mihai Cara (for help, contact `HST Help Desk <https://hsthelp.stsci.edu>`_)

:License: :doc:`LICENSE`

"""
import sys
import os
import abc
import copy
import collections
import tempfile

from astropy.io import fits

from stsci.tools import teal, logutil, textutil, cfgpars, fileutil
from drizzlepac import (adrizzle, ablot, createMedian, drizCR, mdzhandler,
                        processInput, sky, staticMask, util, wcs_functions,
                        __version__ as _drz_version,
                        __version_date__ as _drz_vdate)
import stwcs

from . import __version__, __version_date__


log = logutil.create_logger(__name__, level=logutil.logging.NOTSET)

__all__ = ['Resample', 'Drizzle']


class Resample(abc.ABC):
    """ An abstract class providing interface for resampling and combining
    sets of images onto a rectified frame.

    """
    def __init__(self, config=None, **kwargs):
        self._config = copy.deepcopy(config)
        self._output_sci_data = None
        self._output_wht_data = None
        self._output_ctx_data = None
        self._output_crclean = None
        self._input_file_names = collections.OrderedDict()
        self._reference_image = None
        self._computed_sky = None

    @abc.abstractmethod
    def set_config_parameters(self, **kwargs):
        """ Override individual configuration parameters. """
        pass

    @abc.abstractmethod
    def execute(self):
        """ Run resampling algorithm. """
        pass

    @abc.abstractmethod
    def fast_drop_image(self, drop_file_name):
        """ Re-calculate resampled image using all input images other than
        the one specified by ``drop_file_name``.

        Parameters
        ----------
        drop_file_name : str
            File name of the image to be dropped from the list of input
            images when re-calculating the resampled image.

        """
        pass

    @abc.abstractmethod
    def fast_add_image(self, add_file_name):
        """ Re-calculate resampled image using all input images and adding
        another image to the list of input images specified by
        the ``add_file_name`` parameter.

        Parameters
        ----------
        add_file_name : str
            File name of the image to be added to the input
            image list when re-calculating the resampled image.

        """
        pass

    @property
    def output_sci(self):
        """ Get output file name for output science image or `None`. """
        return self._output_sci_data

    @property
    def output_wht(self):
        """ Get output file name for output weight image or `None`. """
        return self._output_wht_data

    @property
    def output_ctx(self):
        """ Get output file name for context data file or `None`. """
        return self._output_ctx_data

    @property
    def output_crclean(self):
        """ Get file names of the Cosmic Ray (CR) cleaned images (if any). """
        return self._output_crclean

    @property
    def input_image_names(self):
        """ Get an `OrderedDict` of input file names and image extensions or
        `None`.
        """
        return list(self._input_file_names.items())

    @property
    def reference_image(self):
        """ Get/Set Reference image. When ``reference_image`` is `None`,
        output WCS and grid are computed automatically.
        """
        return self._reference_image

    @reference_image.setter
    @abc.abstractmethod
    def reference_image(self, ref_image):
        pass

    @property
    def computed_sky(self):
        return self._computed_sky

    @computed_sky.setter
    def computed_sky(self, computed_sky):
        self._computed_sky = copy.deepcopy(computed_sky)


class Drizzle(Resample):
    """
    """
    taskname = 'astrodrizzle'

    _STEP_STMASK = 'STEP 1: STATIC MASK'
    _STEP_SKYSUB = 'STEP 2: SKY SUBTRACTION'
    _STEP_DRZSEP = 'STEP 3: DRIZZLE SEPARATE IMAGES'
    _STEP_SEPWCS = 'STEP 3a: CUSTOM WCS FOR SEPARATE OUTPUTS'
    _STEP_MEDIAN = 'STEP 4: CREATE MEDIAN IMAGE'
    _STEP_BLOTBK = 'STEP 5: BLOT BACK THE MEDIAN IMAGE'
    _STEP_CRSREJ = 'STEP 6: REMOVE COSMIC RAYS WITH DERIV, DRIZ_CR'
    _STEP_DRZFIN = 'STEP 7: DRIZZLE FINAL COMBINED IMAGE'
    _STEP_FINWCS = 'STEP 7a: CUSTOM WCS FOR FINAL OUTPUT'

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        #self._skyfile_name = None
        #self._skyfile_fid = None # file ID
        #self._skyfile_fh = None # file handle
        self.set_config(config=config, **kwargs)

    #def __copy__(self):
        #cls = self.__class__
        #new_copy_obj = cls.__new__(cls)
        #new_copy_obj.__dict__.update(self.__dict__)

        ## reset temporary skyfile
        #new_copy_obj._tmp_skyfile_name = None
        #new_copy_obj._tmp_skyfile_fid = None
        #new_copy_obj._tmp_skyfile_fh = None

        #return new_copy_obj

    #def __deepcopy__(self, memo):
        #cls = self.__class__
        #new_copy_obj = cls.__new__(cls)
        #memo[id(self)] = new_copy_obj
        #for k, v in self.__dict__.items():
            #setattr(new_copy_obj, k, copy.deepcopy(v, memo))

        ## reset temporary skyfile
        #new_copy_obj._tmp_skyfile_name = None
        #new_copy_obj._tmp_skyfile_fid = None
        #new_copy_obj._tmp_skyfile_fh = None

        #return new_copy_obj

    def set_config_parameters(self, **kwargs):
        self.set_config(config=self._config, **kwargs)

    @property
    def reference_image(self):
        cfg = self._config[self._STEP_FINWCS]
        return cfg['final_refimage'] if cfg['final_wcs'] else None

    @reference_image.setter
    def reference_image(self, ref_image):
        """ Set Reference image. """
        if isinstance(ref_image, str):
            custom_wcs = len(ref_image.strip()) > 0
        else:
            custom_wcs = ref_image is not None

        self.set_config_parameters(
            final_wcs=custom_wcs,
            final_refimage=ref_image,
            final_rot=None,
            final_scale=None,
            final_outnx=None,
            final_outny=None,
            final_ra=None,
            final_dec=None,
            final_crpix1=None,
            final_crpix2=None
        )

    def set_config(self, config=None, **kwargs):
        # Load any user-specified config
        if isinstance(config, str):
            if config == 'defaults':
                # load "TEAL"-defaults (from ~/.teal/):
                config = teal.load(self.taskname)
            else:
                if not os.path.exists(config):
                    raise RuntimeError("Cannot find configuration file '{}'"
                                       .format(config))
                config = teal.load(config, strict=False)
        elif config is None:
            # load 'astrodrizzle' parameter defaults as described in the docs:
            config = teal.load(self.taskname, defaults=True)
        else:
            config = copy.deepcopy(config)

        # If called from interactive user-interface, self._config will not be
        # defined yet, so get defaults using EPAR/TEAL.
        #
        # Also insure that the kwargs (user-specified values) are folded in
        # with a fully populated self._config instance.
        try:
            self._config = util.getDefaultConfigObj(self.taskname, config,
                                                    kwargs, loadOnly=True)
            self._image_names_from_config()

            # initialize computed sky values:
            self._computed_sky = {}
            for fn, extensions in self._input_file_names.items():
                self._computed_sky[fn] = {ext: None for ext in extensions}

            # If user specifies optional parameter for final_wcs specification
            # in kwargs, insure that the final_wcs step gets turned on
            util.applyUserPars_steps(self._config, kwargs, step='3a')
            util.applyUserPars_steps(self._config, kwargs, step='7a')

            log.info("USER INPUT PARAMETERS common to all Drizzle Processing "
                     "Steps:")
            Drizzle._print_cfg(self._config, logfn=log.info)

        except ValueError:
            print("Problem with input parameters. Quitting...",
                  file=sys.stderr)

    @staticmethod
    def _print_key(key, val, lev=0, logfn=print):
        if isinstance(val, dict):
            logfn('\n{}{}:'.format(2*lev*' ', key))
            for kw, vl in val.items():
                Drizzle._print_key(kw, vl, lev+1)
            return
        elif isinstance(val, str):
            logfn("{}{}: '{:s}'".format(2*lev*' ', key, val))
        else:
            logfn("{}{}: {}".format(2*lev*' ', key, val))

    @staticmethod
    def _print_cfg(cfg, logfn):
        if logfn is None:
            logfn = print

        if not cfg:
            logfn('No parameters were supplied')
            return

        keys = cfg.keys()
        smkeys = [k for k in keys if k.islower() and k[0] != '_']
        sectkeys = [k for k in keys if (k.isupper() or k.startswith('STEP '))
                    and k[0] != '_']
        stepkeys = [k for k in sectkeys if k.startswith('STEP ')]
        sectkeys = [k for k in sectkeys if k not in stepkeys]
        for k in smkeys + sectkeys + stepkeys:
            Drizzle._print_key(k, cfg[k], logfn=logfn)
        logfn('')

    def execute(self):
        input_list, output, ivmlist, odict = \
            processInput.processFilenames(self._config['input'])

        if output is None or len(input_list) == 0:
            print(
                textutil.textbox(
                    'ERROR:\nNo valid input files found! Please restart the '
                    'task and check the value for the "input" parameter.'
                ), file=sys.stderr
            )
            return

        stateObj = self._config['STATE OF INPUT FILES']
        procSteps = util.ProcSteps()

        print("AstroDrizzle Version {:s} ({:s}) started at: {:s}\n"
              .format(_drz_version, _drz_vdate, util._ptime()[0]))
        util.print_pkg_versions(log=log)

        filename = self._config.get('runfile', output)
        try:
            if self._config.get('verbose', False):
                util.init_logging(filename, level=util.logging.DEBUG)
            else:
                util.init_logging(filename, level=util.logging.INFO)
        except (KeyError, IndexError, TypeError):
            pass

        imgObjList = None
        try:
            # Define list of imageObject instances and output WCSObject
            # instance based on input paramters
            procSteps.addStep('Initialization')
            imgObjList, outwcs = processInput.setCommonInput(self._config)
            self._image_names_from_imobj(imgObjList)

            # store output names for later use:
            outnam = outwcs.outputNames
            if self._config['build']:
                self._output_sci_data = outnam['outFinal'] + '[SCI,1]'
                self._output_wht_data = outnam['outFinal'] + '[WHT,1]'
                self._output_ctx_data = outnam['outFinal'] + '[CTX,1]'
            else:
                self._output_sci_data = outnam['outSci'] + '[0]'
                self._output_wht_data = outnam['outWeight'] + '[0]'
                self._output_ctx_data = outnam['outContext'] + '[0]'

            procSteps.endStep('Initialization')

            if imgObjList is None or not imgObjList:
                errmsg = "No valid images found for processing!\n"
                errmsg += "Check log file for full details.\n"
                errmsg += "Exiting AstroDrizzle now..."
                print(textutil.textbox(errmsg, width=65))
                print(textutil.textbox(
                    'ERROR:\nAstroDrizzle Version {:s} encountered a problem! '
                    'Processing terminated at {:s}.'
                    .format(_drz_version, util._ptime()[0])
                    ), file=sys.stderr)
                procSteps.reportTimes()
                return

            log.info("USER INPUT PARAMETERS common to all Processing Steps:")
            util.printParams(self._config, log=log)

            # Call rest of MD steps...
            # create static masks for each image
            staticMask.createStaticMask(imgObjList, self._config,
                                        procSteps=procSteps)

            #subtract the sky
            sky.subtractSky(imgObjList, self._config, procSteps=procSteps)
            self._get_computed_sky(imgObjList)

            #drizzle to separate images
            adrizzle.drizSeparate(imgObjList, outwcs, self._config,
                                  wcsmap=None, procSteps=procSteps)

            #create the median images from the driz sep images
            createMedian.createMedian(imgObjList, self._config,
                                      procSteps=procSteps)

            #blot the images back to the original reference frame
            ablot.runBlot(imgObjList, outwcs, self._config, wcsmap=None,
                          procSteps=procSteps)

            #look for cosmic rays
            drizCR.rundrizCR(imgObjList, self._config,
                             procSteps=procSteps)

            if self._config[self._STEP_CRSREJ]['driz_cr_corr']:
                self._output_crclean = [im.outputNames['crcorImage']
                                        for im in imgObjList]
            else:
                self._output_crclean = None

            #Make your final drizzled image
            adrizzle.drizFinal(imgObjList, outwcs, self._config, wcsmap=None,
                               procSteps=procSteps)

            print('\nAstroDrizzle Version {} is finished processing at {}!\n'
                  .format(_drz_version, util._ptime()[0]))

        except:
            print(textutil.textbox(
                'ERROR:\nAstroDrizzle Version {:s} encountered a problem! '
                'Processing terminated at {:s}.'
                .format(_drz_version, util._ptime()[0])),
                  file=sys.stderr)
            procSteps.reportTimes()
            if imgObjList is not None:
                for image in imgObjList:
                    image.close()
            del imgObjList
            del outwcs
            raise

        finally:
            util.end_logging(filename)

        procSteps.reportTimes()

        if imgObjList is not None:
            for image in imgObjList:
                if stateObj['clean']:
                    image.clean()
                image.close()

            del imgObjList
            del outwcs

    def _image_names_from_config(self):
        config = copy.deepcopy(self._config)
        # Interpret input, read and convert and update input files, then
        # return list of input filenames and derived output filename
        asndict, ivmlist, output = processInput.process_input(
            config['input'],
            config['output'],
            updatewcs=False,
            wcskey=config['wcskey'],
            **config['STATE OF INPUT FILES']
        )

        if not asndict:
            self._input_file_names = collections.OrderedDict()
            self._input_wcs = None
            return

        # convert the filenames from asndict into a list of full filenames
        files = [fileutil.buildRootname(f) for f in asndict['order']]
        original_files = asndict['original_file_names']

        # interpret MDRIZTAB, if specified, and update config accordingly
        # This can be done here because MDRIZTAB does not include values
        # for input, output, or updatewcs.
        if 'mdriztab' in config and config['mdriztab']:
            mdriztab_dict = mdzhandler.getMdriztabParameters(files)

            # Update self._config with values from mpars
            cfgpars.mergeConfigObj(config, mdriztab_dict)

        imname = collections.OrderedDict()
        auto_group = config['group'] is None or config['group'].strip() == ''

        sky = {}

        for f in files:
            image = processInput._getInputImage(f, group=config['group'])
            if auto_group:
                extvers = list(range(1, image._numchips + 1))
            else:
                extvers = image.group

            imname[f] = [(image.scienceExt, i) for i in extvers]

            image.close()
            del image

        self._input_file_names = imname

    def _image_names_from_imobj(self, imobj):
        config = self._config

        imname = collections.OrderedDict()
        auto_group = config['group'] is None or config['group'].strip() == ''

        for image in imobj:
            if auto_group:
                extvers = list(range(1, image._numchips + 1))
            else:
                extvers = image.group

            imname[image._filename] = [(image.scienceExt, i) for i in extvers]

        self._input_file_names = imname

    def _get_computed_sky(self, imobj):
        """ Return a dictionary of computed sky values. """
        sky = {}
        config = self._config
        auto_group = config['group'] is None or config['group'].strip() == ''

        for image in imobj:
            sky[image._filename] = {}
            if auto_group:
                extvers = list(range(1, image._numchips + 1))
            else:
                extvers = image.group
            ext = [(image.scienceExt, i) for i in extvers]
            sky[image._filename] = {e: image[e].computedSky for e in ext}

        self._computed_sky = sky

    def fast_drop_image(self, drop_file_name):
        """ Re-calculate resampled image using all input images other than
        the one specified by ``drop_file_name``.

        Parameters
        ----------
        drop_file_name : str
            File name of the image to be dropped from the list of input
            images when re-calculating the resampled image.

        """
        self.fast_replace_image(drop_file_name=drop_file_name,
                                add_file_name=None)


    def fast_add_image(self, add_file_name):
        """ Re-calculate resampled image using all input images and adding
        another image to the list of input images specified by
        the ``add_file_name`` parameter.

        Parameters
        ----------
        add_file_name : str
            File name of the image to be added to the input
            image list when re-calculating the resampled image.

        """
        self.fast_replace_image(drop_file_name=None,
                                add_file_name=add_file_name)

    def fast_replace_image(self, drop_file_name, add_file_name):
        """ Re-calculate resampled image using all input images and adding
        another image to the list of input images specified by
        the ``add_file_name`` parameter.

        Parameters
        ----------
        add_file_name : str
            File name of the image to be added to the input
            image list when re-calculating the resampled image.

        """
        if add_file_name is None and drop_file_name is None:
            # nothing to do...
            return

        sta = None if add_file_name is None else os.stat(add_file_name)
        std = None if drop_file_name is None else os.stat(drop_file_name)

        if sta == std:
            # Both files are identical. Nothing to do...
            return

        fnames = list(self._input_file_names.keys())
        modified = True

        if drop_file_name is not None:
            dropping = True
            new_fnames = [f for f in fnames if os.stat(f) != std]
            if len(new_fnames) == len(fnames):
                # file not found in input list. Nothing to do (or drop):
                dropping = False
            else:
                fnames = new_fnames

        if add_file_name is not None:
            adding = True
            for f in fnames:
                if os.stat(f) == sta:
                    # file already in input list. Nothing to do...
                    adding = False
                    break
            if adding:
                fnames.append(add_file_name)

        # update input image list
        input_files = ','.join(fnames)
        self._config['input'] = input_files

        # re-calculate resampled image
        config = copy.deepcopy(self._config)
        try:
            skyfile = config[self._STEP_SKYSUB]['skyfile']
            skyfile = '' if skyfile is None else skyfile.strip()
            if config[self._STEP_SKYSUB]['skysub'] and skyfile == '':
                tmpf = tempfile.NamedTemporaryFile(
                    mode='w+t', suffix='.txt', prefix='tmp_skyfile_', dir='./'
                )
                self._create_skyfile(tmpf.file)
                skyfile = tmpf.name
                # set sky-related config parameters:
                self._config[self._STEP_SKYSUB]['skyuser'] = ''
                self._config[self._STEP_SKYSUB]['skyfile'] = skyfile

            else:
                tmpf = None

            # turn off steps that can be skipped for extra speed:
            self._config[self._STEP_DRZSEP]['driz_separate'] = False
            self._config[self._STEP_MEDIAN]['median'] = False
            self._config[self._STEP_BLOTBK]['blot'] = False
            self._config[self._STEP_CRSREJ]['driz_cr'] = False
            self._config[self._STEP_DRZFIN]['driz_combine'] = True

            # keep the same output image size & WCS:
            if not self._config[self._STEP_FINWCS]['final_wcs']:
                self._config[self._STEP_FINWCS]['final_wcs'] = True
                self._config[self._STEP_FINWCS]['final_refimage'] = self.output_sci

            self.execute()

        except:
            raise

        finally:
            # restore config
            self._config = config

            # close temporary skyfile:
            if tmpf is not None:
                tmpf.close()

    def _create_skyfile(self, skyfile):
        self._image_names_from_config()
        sky_kwd = 'MDRIZSKY'

        skyfile.seek(0)

        for fn in self._input_file_names.keys():
            extensions = self._input_file_names[fn]
            with fits.open(fn) as h:
                skyvalues = []
                for ext in extensions:
                    if sky_kwd in h[ext].header:
                        skyvalues.append(h[ext].header[sky_kwd])
                    else:
                        raise ValueError(
                            "Missing '{}' value for file '{}'[{:s},{:d}]"
                            .format(sky_kwd, ext[0], ext[1]))

                line = (
                    "{:s}\t" + '\t'.join(len(skyvalues) * ['{:.15g}']) + '\n'
                ).format(fn, *skyvalues)
                skyfile.write(line)
        skyfile.truncate()
        skyfile.flush()
        skyfile.seek(0)
