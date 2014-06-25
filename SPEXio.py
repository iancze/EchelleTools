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

parser.add_argument("--npy", action="store_true", help="Write out as many *.npy files.")
parser.add_argument("--hdf", action="store_true", help="Write out as a single hdf5 file.")
parser.add_argument("--clip", type=int, help="Truncate the data file after this wavelengeth (AA)")
parser.add_argument("--air", action="store_true", help="Shift from vacuum to air wavelengths")

parser.add_argument("--interpolate", action="store_true", help="Interpolate atmospheric chunks with a line, "
                                                               "and create mask.")

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

def vacuum_to_air(wl):
    '''
    Converts vacuum wavelengths to air wavelengths using the Ciddor 1996 formula.

    :param wl: input vacuum wavelengths
    :type wl: np.array

    :returns: **wl_air** (*np.array*) - the wavelengths converted to air wavelengths

    .. note::

        CA Prieto recommends this as more accurate than the IAU standard.'''

    sigma = (1e4 / wl) ** 2
    f = 1.0 + 0.05792105 / (238.0185 - sigma) + 0.00167917 / (57.362 - sigma)
    return wl / f


class SPEXProcessor:
    def __init__(self, txt_fn, out_fn):

        # self.norders = 51
        self.txt_fn = txt_fn
        self.out_fn = out_fn #Directory which to place all output products

    def process_all(self, npy=True, hdf=True):

        #Use ascii to read in wl, fl, and sigma
        data = ascii.read(self.txt_fn, names=["wl", "fl", "sigma"])
        wl = data['wl'] * 1e4 # convert from microns to angstroms
        fl = data['fl'] * 0.1 # convert from W/m^2/micron to erg/s/cm^2/Ang
        sigma = data['sigma'] * 0.1 # convert from W/m^2/micron to erg/s/cm^2/Ang

        #Any region with flux less than 0 should be masked
        masks = (fl >= 0)

        if args.clip is not None:
            #If segment, then just clip.
            ind = np.sum(wl <= args.clip)
            wl = wl[:ind]
            fl = fl[:ind]
            sigma = sigma[:ind]
            masks = masks[:ind]


        if args.interpolate:
            # Find extended gap in the data and linearly interpolate across them

            #What is the threshold that defines a "gap"?
            # print(np.sort(np.diff(wl))[::-1][:20])

            #Say 20 angstroms
            ind_gaps = np.argwhere(np.diff(wl) > 20.)
            for i in range(len(ind_gaps)):
                #Re-find the first gap, since we will be updating wl in this loop
                gap = np.argwhere(np.diff(wl) > 20)[0]

                start = wl[gap]
                end = wl[gap + 1]

                #Use the same wl spacing as the nearest pixels
                spacing0 = wl[gap] - wl[gap - 1]
                spacing1 = wl[gap + 2] - wl[gap + 1]
                spacing = 0.5 * (spacing0 + spacing1)
                npoints = np.rint((end - start)/spacing).astype(int)

                wl_fill = np.linspace(start + spacing, end - spacing, npoints - 2)
                fl_fill = -99.9 * np.ones_like(wl_fill)
                sigma_fill = -99.9 * np.ones_like(wl_fill)
                mask_fill = ~np.ones_like(wl_fill, dtype='bool')

                wl = np.insert(wl, gap + 1, wl_fill)
                fl = np.insert(fl, gap + 1, fl_fill)
                sigma = np.insert(sigma, gap + 1, sigma_fill)
                masks = np.insert(masks, gap + 1, mask_fill)

        if args.air:
            wl = vacuum_to_air(wl)

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

            mask_data = hdf5.create_dataset("masks", shape, dtype="b", compression='gzip', compression_opts=9)
            mask_data[:] = masks

            hdf5.close()

        if npy:
            np.save(self.out_fn + ".wls.npy",wl)
            np.save(self.out_fn + ".fls.npy",fl)
            np.save(self.out_fn + ".sigmas.npy", sigma)
            np.save(self.out_fn + ".masks.npy", masks)

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
