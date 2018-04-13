
"""
A module that manages drizzling input images into a combined image and also
"inverse" blotting.

:Author: Mihai Cara (contact: help@stsci.edu)

:License: :doc:`../LICENSE`

"""
from __future__ import (absolute_import, division, unicode_literals,
                        print_function)

import copy
import sys
import abc

import six

from stsci.tools import teal, logutil, textutil, cfgpars, fileutil
from drizzlepac import (adrizzle, ablot, createMedian, drizCR, mdzhandler,
                        processInput, sky, staticMask, util, wcs_functions)
import stwcs

from . import __version__, __version_date__


log = logutil.create_logger(__name__)

__all__ = ['Resample', 'Drizzle']


@six.add_metaclass(abc.ABCMeta)
class Resample():
    """ An abstract class providing interface for resampling and combining
    sets of images onto a rectified frame.

    """
    def __init__(self, config=None, **kwargs):
        self._config = copy.deepcopy(config)
        self._output_sci_data = None
        self._output_wht_data = None
        self._output_ctx_data = None
        self._output_wcs = None
        self._input_image_names = None

    @abc.abstractmethod
    def execute(self):
        """ Run resampling algorithm. """
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
    def output_wcs(self):
        """ Get ``WCS`` object of the output image or `None`. """
        return self._output_wcs

    @property
    def input_image_names(self):
        """ Get a list of input file names or `None`. """
        return self._input_file_names


class Drizzle(Resample):
    """
    """
    taskname = 'astrodrizzle'

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.set_config(config=config, **kwargs)

    def set_config(self, config=None, **kwargs):
        # Load any user-specified config
        if isinstance(config, six.string_types):
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
        sectkeys = [k for k in keys if k.isupper() and k[0] != '_']
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
              .format(__version__, __version_date__, util._ptime()[0]))
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
                    'ERROR:\nAstroDrizzle Version %s encountered a problem!  '
                    'Processing terminated at %s.' %
                    (__version__, util._ptime()[0])), file=sys.stderr)
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

            #Make your final drizzled image
            adrizzle.drizFinal(imgObjList, outwcs, self._config, wcsmap=None,
                               procSteps=procSteps)

            print('\nAstroDrizzle Version {} is finished processing at {}!\n'
                  .format(__version__, util._ptime()[0]))

        except:
            print(textutil.textbox(
                'ERROR:\nAstroDrizzle Version {:s} encountered a problem!  '
                'Processing terminated at {:s}.'
                .format(__version__, util._ptime()[0])), file=sys.stderr)
            procSteps.reportTimes()
            if imgObjList is not None:
                for image in imgObjList:
                    image.close()
                del imgObjList
                del outwcs

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
            self._input_image_names = None
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

        imname = []
        for f in files:
            image = processInput._getInputImage(f, group=config['group'])
            if config['group'] is None or config['group'].strip() == '':
                extvers = range(1, image._numchips + 1)
            else:
                extvers = image.group

            for i in extvers:
                fn = '{:s}[{:s},{:d}]'.format(f, image.scienceExt, i)
                imname.append(fn)

            image.close()
            del image

        self._input_file_names = imname

    def _image_names_from_imobj(self, imobj):
        config = self._config
        imname = []

        for image in imobj:
            if config['group'] is None or config['group'].strip() == '':
                extvers = range(1, image._numchips + 1)
            else:
                extvers = image.group

            for i in extvers:
                fn = '{:s}[{:s},{:d}]'.format(image._filename, image.scienceExt, i)
                imname.append(fn)

        self._input_file_names = imname
