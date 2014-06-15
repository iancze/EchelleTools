#!/usr/bin/env python

'''
Process a IRTF/SPEX spectrum into an HDF5 file. Command-line program which takes arguments text file and creates
output file.

SPEX data have been shifted to zero radial velocity and are in vacuum wavelengths.

Products in the HDF5 file are:

* wl cube
* fl cube
* sigma array (based off of un-blazed-corrected spectrum, normed to median of order level)

'''

import argparse
parser = argparse.ArgumentParser(description="Process SPEX echelle spectra into a numpy or HDF5 data cube.")
parser.add_argument("txtfile", help="The text file containing the data.")
parser.add_argument("outfile", help="Output Filename to contain the processed file."
                                    "Should have no extension, *.hdf5 or *.npy added automatically.")

parser.add_argument("--orders", action="store_true", help="Break the spectrum into multiple orders.")
parser.add_argument("--npy", action="store_true", help="Write out as many *.npy files.")
parser.add_argument("--hdf", action="store_true", help="Write out as a single hdf5 file.")
# parser.add_argument("-t", "--trim", type=int, default=6, help="How many pixels to trim from the front of the file. "
#                                                               "Default is 6")

# The orders argument could be implemented by doing np.diff(), searching for the gaps over some threshold (100 ang?)

# Units: do we want to convert from microns to ang?

parser.add_argument("--clobber", action="store_true", help="Overwrite existing outfile?")

args = parser.parse_args()

import numpy as np
from astropy.io import ascii
import os
import h5py

c_ang = 2.99792458e18 #A s^-1
c_kms = 2.99792458e5 #km s^-1

#n @ 3000: 1.0002915686329712
#n @ 6000: 1.0002769832562917
#n @ 8000: 1.0002750477973053

n_air = 1.000277
c_ang_air = c_ang/n_air
c_kms_air = c_kms/n_air

class SPEXProcessor:
    def __init__(self, txt_fn, out_fn):

        # self.norders = 51
        self.txt_fn = txt_fn
        self.out_fn = out_fn #Directory which to place all output products

    def process_all(self, npy=True, hdf=True):

        #Use ascii to read in wl, fl, and sigma
        data = ascii.read(self.txt_fn, names=["wl", "fl", "sigma"])
        wl = data['wl'] * 1e4 # convert from microns to angstroms
        fl = data['fl']
        sigma = data['sigma']

        if hdf:
            #Create HDF5 file with basename
            hdf5 = h5py.File(self.out_fn + ".hdf5", "w")

            shape = wl.shape

            wl_data = hdf5.create_dataset("wls", shape, dtype="f8", compression='gzip', compression_opts=9)
            wl_data[:] = wl

            fl_data = hdf5.create_dataset("fls", shape, dtype="f", compression='gzip', compression_opts=9)
            fl_data[:] = fl


            sig_data = hdf5.create_dataset("sigmas", shape, dtype="f", compression='gzip', compression_opts=9)
            sig_data[:] = sigma

            hdf5.close()

        if npy:
            np.save(self.out_fn + ".wls.npy",wl)
            np.save(self.out_fn + ".fls.npy",fl)
            np.save(self.out_fn + ".sigmas.npy", sigma)

def main():
    #Check to see if outfile exists. If --clobber, overwrite, otherwise exit.
    if os.path.exists(args.outfile):
        if not args.clobber:
            import sys
            sys.exit("Error: outfile already exists and --clobber is not set. Exiting.")

    #assert that we actually specified the file extensions correctly
    if (".txt" not in args.txtfile.lower()):
        import sys
        sys.exit("Must specify *.txt files. See --help for more details.")

    #Create the SPEXProcessor using the command line arguments
    SP = SPEXProcessor(args.txtfile, args.outfile)

    #Do the processing
    if args.npy:
        SP.process_all(npy=args.npy, hdf=False)
    if args.hdf:
        SP.process_all(npy=False, hdf=args.hdf)
    else:
        SP.process_all()


if __name__ == "__main__":
    main()
