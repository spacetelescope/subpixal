.. _release_notes:

=============
Release Notes
=============


.. subpixal (unreleased)
   =====================


0.1.1.dev (15-December-2019)
============================

- Fixed incorrect parameter being passed to SExtractor. [#47]


0.1.0 (30-September-2019)
=========================

- Added ``combine_seg_mask`` argument to ``align.align_images()`` and
  other functions that allows users to turn off combining segmentation mask
  with other DQ masks for the cutouts. Practical application of this
  option is to turn off zeroing of pixels that are outside of the
  segmentation mask in the blotted cutouts. [#44]

- Added support for zero-normalized cross-correlation (ZNCC) and normalized
  cross-correlation (NCC) algorithms. [#42]

- Allow alignment code to run with without cosmic ray-cleaned images. [#41]

- Reliability enhancement in handling cases when sky computation is
  turned off. [#40]

- Modified the formula for computing ``RMSE`` of the fit in *image pixels*
  to take into account weights when available. [#39]


0.0.5 (22-February-2019)
========================

- Added support for weighted fitting. Added parameter ``'use_weights'``
  that can be used to enable/disable weighted fitting when input catalogs
  have a column called ``'weight'``. [#38]


0.0.4 (03-January-2019)
=======================

- Added support for keeping top/bottom number of sources according to
  values in a specified catalog's column. [#37]

- The direction of the displacement as well as the direction of the fit
  have been reversed (bug fix). [#36]

- Instead of reporting ``XRMS`` and ``YRMS`` (rms of the fit in the tangent
  plane; i.e., the RMS displacement of the image source positions wrt.
  reference source positions, now the code will report total RMS ``FIT_RMS``
  computed as ``sqrt(XRMS**2+YRMS**2)`` and `IMG_RMS` (equivalent of
  ``FIT_RMS`` but computed in input image pixels - hence the problem with this
  measure for images affected by distortion). [#36]

- Added a parameter (``wcsupdate``) that allows a choice of when to update
  image headers with an aligned WCS: as soon as an image is fit (and then it
  can be used by next images) or wait until the end of the iteration and update
  all images at once. [#36]


0.0.3 (27-December-2018)
========================

- Make sure ``execute()`` is called before returning segmentation
  image data. [#32]

- Add missing import. [#32]

- Setup dependency clean-up. [#31]

- Fix changelog. [#30]


0.0.2 (23-December-2018)
========================

- Initial fully operational release. [#29]


0.0.1 (10-April-2018)
=====================

- Initial release. [#1]
