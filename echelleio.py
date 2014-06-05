#!/usr/bin/env python2

import numpy as np
from pyraf import iraf
import tempfile
import os
from astropy.io import fits
import h5py

'''
Process a TRES spectrum into an HDF5 file. Command-line program which takes arguments for raw file, calibrated file,
and output file.

Products in the HDF5 file are:

* wl cube
* fl cube
* sigma array (based off of un-blazed-corrected spectrum, normed to median of order level).
* mask array (if specified)

'''

import argparse
parser = argparse.ArgumentParser(description="Process TRES echelle spectra into an HDF5 data cube, automatically "
                                             "doing the barycentric correction.")
parser.add_argument("rawfile", help="The un-blaze-corrected, un-flux-calibrated FITS file.")
parser.add_argument("calfile", help="The blaze-corrected, flux-calibrated FITS file.")
parser.add_argument("outfile", help="Output HDF file to contain the processed file.")

parser.add_argument("-t", "--trim", type=int, default=6, help="How many pixels to trim from the front of the file.")
parser.add_argument("--noBCV", action="store_true", help="If provided, don't do the barycentric correction.")

parser.add_argument("--clobber", action="store_true", help="Overwrite existing outfile?")

c_ang = 2.99792458e18 #A s^-1
c_kms = 2.99792458e5 #km s^-1

#n @ 3000: 1.0002915686329712
#n @ 6000: 1.0002769832562917
#n @ 8000: 1.0002750477973053

n_air = 1.000277
c_ang_air = c_ang/n_air
c_kms_air = c_kms/n_air

#Determine the relative path to shorten calls to wspectext
#dir = os.path.relpath(config['dir']) + "/"

class TRESProcessor:
    def __init__(self, raw_fn, cal_fn, out_fn, trim=6, BCV_cor=True):

        self.raw_fn = raw_fn
        self.cal_fn = cal_fn
        self.out_fn = out_fn #Directory which to place all output products

        self.trim = trim
        self.BCV_cor = BCV_cor

        self.raw_dir = tempfile.mkdtemp() #Temporary directory to output IRAF products
        self.cal_dir = tempfile.mkdtemp()

        #Read BCV from header
        hdulist = fits.open(self.raw_fn)
        head = hdulist[0].header
        try:
            self.BCV = head["BCV"]
        except KeyError:
            self.BCV = None
        hdulist.close()


    def wechelletxt(self, infile, outdir):
        for i in range(1, 52): #Do this for all 51 orders
            inp = infile + "[*,{:d}]".format(i)
            out = outdir + "/{:0>2d}.txt".format(i)
            iraf.wspectext(input=inp, output=out)

    def rechellenpflat(self, bname):
        '''Reads text files. Returns two 2D numpy arrays of shape (norders, len_wl).
        The first is wl, the second fl. For example, GWOri has shape (51,2304). Assumes each order is the same
        length.'''
        inp = bname + "/01.txt"
        wl, fl = np.loadtxt(inp, unpack=True)
        len_wl = len(wl)
        wls = np.empty((self.norders, len_wl))
        fls = np.empty((self.norders, len_wl))

        for i in range(self.norders):
            inp = bname + "/{:0>2d}.txt".format(i + 1)
            wl, fl = np.loadtxt(inp, unpack=True)
            wls[i] = wl
            fls[i] = fl
        return [wls, fls]

    def write_dirs(self):
        #Write the files
        self.wechelletxt(self.raw_fn, self.raw_dir)
        self.wechelletxt(self.cal_fn, self.cal_dir)

    # def process_all_npy(self):
    #     #Use IRAF/wspectxt to create temporary text files
    #     self.write_dirs()
    #
    #     #read files back into numpy arrays
    #     wlsb, flsb = self.rechellenpflat(self.raw_dir)
    #     wlsf, flsf = self.rechellenpflat(self.cal_dir)
    #
    #     #Trim all files
    #     wlsb = wlsb[:,self.trim:]
    #     flsb = flsb[:,self.trim:]
    #     wlsf = wlsf[:,self.trim:]
    #     flsf = flsf[:,self.trim:]
    #
    #     #Do Barycentric correction on files?
    #     if (self.BCV is not None) and self.BCV_cor:
    #         wlsb = wlsb * np.sqrt((c_kms_air + self.BCV) / (c_kms_air - self.BCV))
    #         wlsf = wlsf * np.sqrt((c_kms_air + self.BCV) / (c_kms_air - self.BCV))
    #
    #     #write flux files to .npy
    #     np.save(self.outpath + ".wls.npy",wlsf)
    #     np.save(self.outpath + ".fls.npy",flsf)
    #
    #     #create sigma file
    #     #set where (cts == 0) to something small
    #     flsb[flsb==0] = 0.001
    #     flsf[flsf==0] = 1e-18
    #     noise_to_signal = 1./np.sqrt(np.abs(flsb))
    #     sigma = noise_to_signal * flsf
    #     np.save(self.outpath + ".sigmas.npy", sigma)

    def process_all_hdf5(self):
        #Use IRAF/wspectxt to create temporary text files
        self.write_dirs()

        #read files back into numpy arrays
        wlsb, flsb = self.rechellenpflat(self.raw_dir)
        wlsf, flsf = self.rechellenpflat(self.cal_dir)

        #Trim all files
        wlsb = wlsb[:,self.trim:]
        flsb = flsb[:,self.trim:]
        wlsf = wlsf[:,self.trim:]
        flsf = flsf[:,self.trim:]

        #Do Barycentric correction on files?
        if (self.BCV is not None) and self.BCV_cor:
            wlsb = wlsb * np.sqrt((c_kms_air + self.BCV) / (c_kms_air - self.BCV))
            wlsf = wlsf * np.sqrt((c_kms_air + self.BCV) / (c_kms_air - self.BCV))

        #Create HDF5 file with basename
        # self.outpath
        hdf5 = h5py.File(self.out_fn, "w")
        hdf5.attrs["BCV"] = self.BCV

        shape = wlsb.shape

        wl_data = hdf5.create_dataset("wls", shape, dtype="f8", compression='gzip', compression_opts=9)
        wl_data[:] = wlsf

        fl_data = hdf5.create_dataset("fls", shape, dtype="f", compression='gzip', compression_opts=9)
        fl_data[:] = flsf

        #create sigma file
        #set where (cts == 0) to something small
        flsb[flsb==0] = 0.001
        flsf[flsf==0] = 1e-18
        noise_to_signal = 1./np.sqrt(np.abs(flsb))
        sigma = noise_to_signal * flsf
        sig_data = hdf5.create_dataset("sigmas", shape, dtype="f", compression='gzip', compression_opts=9)
        sig_data[:] = sigma

        hdf5.close()

def main():
    args = parser.parse_args()

    #Check to see if outfile exists. If --clobber, overwrite, otherwise exit.
    if os.path.exists(args.outfile):
        if not args.clobber:
            import sys
            sys.exit("Error: outfile already exists and --clobber is not set. Exiting.")
    else:
        os.makedirs(args.outdir)

    #assert that we actually specified the file extensions correctly
    if (".fit" not in args.rawfile.lower()) or (".fit" not in args.calfile.lower()) or (".hdf5" not in args.outfile.lower()):
        import sys
        sys.exit("Must specify *.fits and *.hdf5 files. See --help for more details.")

    #Create the TRESProcessor using the command line arguments
    TP = TRESProcessor(args.rawfile, args.calfile, args.outfile, trim=args.trim, BCV_cor=(not args.noBCV))
    TP.process_all_hdf5()


if __name__ == "__main__":
    main()
