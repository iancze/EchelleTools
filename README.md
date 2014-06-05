EchelleTools
============

Command-line tools to process high resolution echelle spectra from astronomical instruments (TRES, HIRES, Spex, etc...) into a common format.

This is a collection of hacked-together scripts designed as a stop-gap measure to get my spectra out of the FITS format as and into numpy arrays/HDF5 as quickly as possible, so that I can get on with my analysis with python. It pushes all of the complicated intricacies of FITS reading into an IRAF dependency, so if you don't already have it installed... good luck with that.

Those wishing for a more sophisticated, purely pythonic interface for reading FITS spectra should check out astropy/specutils.
