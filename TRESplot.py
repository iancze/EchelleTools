#!/usr/bin/env python
import argparse
parser = argparse.ArgumentParser(prog="TRESplot.py",
                                 description="You've already run TRESio, now lets plot all the spectra!")
parser.add_argument("file", help="The HDF5 file you want to plot.")
parser.add_argument("--orders", default="all", help="Which orders of the spectrum do you want to plot?")
parser.add_argument("--spec2", help="The second spectrum to plot")
parser.add_argument("--norm", action="store_true", help="Normalize each order to 1?")

args = parser.parse_args()

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter as FSF
from Starfish.spectrum import DataSpectrum

spec = DataSpectrum.open(args.file, orders=args.orders)

if args.spec2:
    spec2 = DataSpectrum.open(args.spec2, orders=args.orders)


#Set up the plot.
width = 20. #in; the size of my monitor
h = 1.5 #height per order.
norders = len(spec.wls)

matplotlib.rc("font", size=8)

fig, ax = plt.subplots(nrows=norders, figsize=(width, h*norders))

for a, wl, fl, sigma, order in zip(ax, spec.wls, spec.fls, spec.sigmas, spec.orders):
    if args.norm:
        fl /= np.average(fl)
    a.plot(wl, fl)
    #a.plot(wl, sigma)
    a.set_ylabel("Order {}".format(order + 1))
    a.set_xlim(wl[0], wl[-1])
    a.xaxis.set_major_formatter(FSF("%.0f"))

if args.spec2:
    for a, wl, fl in zip(ax, spec2.wls, spec2.fls):
        if args.norm:
            fl /= np.average(fl)
        a.plot(wl, fl, "r")


ax[-1].set_xlabel(r"$\lambda$ [\AA]")

fig.subplots_adjust(left=0.04, right=0.99, bottom=0.01, top=0.99, hspace=0.2)
fig.savefig("out.pdf")