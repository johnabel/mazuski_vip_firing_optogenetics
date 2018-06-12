"""
j.h.abel 19/7/2016

adjusting single-step for multi-step optimization
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


def wavelet_ridge(times, cwtmatr, cwt_extent):
    """
    returns the ridge values (instantaneous period) from the continuous wavelet
    output
    """
    abss = np.abs(cwtmatr)
    ridge_locs = np.argmax(abss, axis=0)
    argmax_val = cwt_extent[2]
    argmin_val = cwt_extent[3]
    scaled_ridge = (ridge_locs/abss.shape[0])*(argmax_val-argmin_val) + \
                                argmin_val
    ridge_spl = UnivariateSpline(times, scaled_ridge, s=len(times)*2)
    return scaled_ridge, ridge_locs
    

bin_param = 15
times, counts = load_invivo_data(example_path, binning = bin_param)
widths = np.arange(15*(60/bin_param), 28*(60/bin_param), 0.25*(60/bin_param))


#plt.figure()
#plt.plot(times, counts, 'k')

#pywt version
cwtmatr, freqs = pywt.cwt(counts, widths, 'morl')
periods = 1/freqs/(60/bin_param)
cwt_extent = [np.min(times)/24., np.max(times)/24., 
             np.max(periods), np.min(periods)]
ridge, ridge_locs = wavelet_ridge(times, cwtmatr, cwt_extent)

plt.figure()
plt.imshow(np.abs(cwtmatr), aspect='auto', extent=cwt_extent,
            vmax=np.abs(cwtmatr).max(), vmin=-np.abs(cwtmatr).max())
plt.plot(times/24., ridge, 'k')
#plt.xlim([3,16])
#plt.ylim([20,32])
plt.title("Period")


plt.figure()
plt.imshow(np.real(cwtmatr), aspect='auto', extent=cwt_extent,
            vmax=np.real(cwtmatr).max(), vmin=np.real(cwtmatr).min())
plt.plot(times/24., ridge, 'k')
#plt.xlim([3,16])
#plt.ylim([20,32])
plt.title("Phase")













































