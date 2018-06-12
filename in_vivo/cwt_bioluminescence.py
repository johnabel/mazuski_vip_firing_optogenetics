"""
j.h.abel 25 Jan 2017

Using the cwt code based on Tanya Leise's code for analyzing rhythms of wheel 
running.
"""

#
#
# -*- coding: utf-8 -*-
#basic packages
from __future__ import division
import sys
import os

#3rd party packages
import numpy as np
from scipy import stats, signal
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from matplotlib import gridspec
import Bioluminescence as bl
import pywt 

example_path = 'data/examples/Channel1_122616.csv'

def load_invivo_data(path, binning=None):
    """
    loads the wheel running rhythms from a csv located at path
    if binning is int, bins into that many minute units
    """
    # load the data and assemble parts
    data = np.loadtxt(path, delimiter=',', dtype=str)
    counts = data[1:,3]
    recordings = [loc for loc in range(len(counts)) if counts[loc].isdigit()]
    days   = data[1:,0].astype(float)
    hours  = data[1:,1].astype(float)
    mins   = data[1:,2].astype(float)
    times  = np.asarray([days[i]*24+(hours[i]+mins[i]/60)
                    for i in range(len(days))])
    
    # get into format times, counts
    ret_times = times[recordings]
    ret_counts = counts[recordings].astype(float)
    
    # if binning
    if type(binning) is int:
        new_times = times[::binning]
        bins = np.hstack([new_times,times[-1]])
        digitized = numpy.digitize(ret_times, bins)
        new_counts = np.array([ret_counts[digitized == i].sum() 
                        for i in range(1, len(bins))])
        # re-assign                    
        ret_counts = new_counts
        ret_times = new_times
    
    return ret_times, ret_counts


times, counts = load_invivo_data(example_path, binning = None)

wavres = bl.continuous_wavelet_transform(times, counts, 
                                         shortestperiod=5, longestperiod = 32)





plt.figure()
plt.pcolormesh(wavres['x']/24, wavres['tau'],wavres['cwt_abs'], 
          vmax=wavres['cwt_abs'].max(), vmin=wavres['cwt_abs'].min())
plt.plot(wavres['x']/24, wavres['period'], 'k')
plt.title("Magnitude")
plt.xlabel("Time (day)")
plt.ylabel(r"$\tau$ (h)")
plt.tight_layout(**plo.layout_pad)

# plotting the phase progression
plt.figure()
plt.pcolormesh(wavres['x']/24, wavres['tau'],np.real(wavres['cwt']), 
            vmax=np.real(wavres['cwt']).max(), vmin=np.real(wavres['cwt']).min())
plt.plot(wavres['x']/24, wavres['period'], 'k')
plt.title("Phase")
plt.xlabel("Time (day)")
plt.ylabel(r"$\tau$ (h)")
plt.tight_layout(**plo.layout_pad)


















