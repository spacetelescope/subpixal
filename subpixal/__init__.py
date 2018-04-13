""" ``subpixal``: A package for aligning images and correcting their ``WCS``
using sub-pixel cross-correlation algorithm developed by Andrew Fruchter and
Rebekah Hounsell.

"""

from __future__ import (absolute_import, division, unicode_literals,
                        print_function)
import os


__docformat__ = 'restructuredtext en'
__author__ = 'Mihai Cara'


from .version import *

from .catalogs import *
from .resample import *
from .utils import *
