"""
j.h.abel 25 Jan 2017

Using the cwt code based on Tanya Leise's code for analyzing rhythms of wheel 
running. Follows the ideas in 
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
import matplotlib.patches as patches
from matplotlib import gridspec

import PlotOptions as plo
import Bioluminescence as bl


import pywt 


def load_invivo_data(path, binning=None):
    """
    loads the wheel running rhythms from a csv located at path
    if binning is int, bins into that many minute units
    """
    # load the data and assemble parts
    data = np.loadtxt(path, delimiter=',', dtype=str)
    counts = data[3:,3]
    recordings = [loc for loc in range(len(counts)) if counts[loc].isdigit()]
    days   = data[3:,0].astype(float)
    hours  = data[3:,1].astype(float)
    mins   = data[3:,2].astype(float)
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
    
    stimulations = data[:2,:4]
    
    return {'times'             : ret_times, 
            'counts'            : ret_counts,
            stimulations[0,0]   : stimulations[0, 1:].astype(float),
            stimulations[1,0]   : stimulations[1, 1:].astype(float),
            'path'              : path
            }

def find_cwt0_region(region, x, mhw):
    """ finds the zero-cross of the mexican hat wavelet, as suggested in
    Leise 2013. """
    specific_region = mhw[region[0]:region[1]]
    specific_x_locs = np.arange(len(specific_region))+region[0]
    specific_times = x[specific_x_locs]
    
    # find peaks
    zeros_loc = numpy.where(numpy.diff(numpy.sign(specific_region)))[0]
    # spline interpolate for simplicity
    spl = UnivariateSpline(specific_times, specific_region, k=3, s=0)
    
    onset_time =spl.roots()
    if len(onset_time)>0:
        return onset_time[0]



def cwt_onset(times, counts):
    """ identifies wheel running onset from times and counts"""
    x = times
    y = counts
    bin_param = 15
    widths = np.arange(3*(60/bin_param), 
                       9*(60/bin_param), 
                       0.05*(60/bin_param))
    
    #pywt version
    cwtmatr, freqs = pywt.cwt(y, widths, 'mexh')
    periods = 1/freqs/(60/bin_param)
    
    inst_per_loc = np.argmax(np.abs(cwtmatr.T),1)
    inst_ampl = np.asarray([cwtmatr.T[idx, loc] 
                            for idx,loc in enumerate(inst_per_loc)])
    

    # identify regions of increasing activity
    maxs = signal.argrelextrema(inst_ampl, np.greater, order=40)[0]
    mins = signal.argrelextrema(inst_ampl, np.less, order=40)[0]
    
    # find the 0-crosses here
    inc_regions = []
    for mini in mins:
        try:
            maxi = np.min(maxs[np.where(maxs>mini)])
            inc_regions.append([mini,maxi])
        except ValueError:
            # if there is no following max
            pass
    
    # get the onset times
    onsets = []
    for region in inc_regions[:-1]:
        onset = find_cwt0_region(region,x,inst_ampl)
        if onset is not None:
            onsets.append(onset)
    
    
    return_data = {
        'x'         : x,
        'y'         : y,
        'onsets'    : np.array(onsets),
        'dwt'       : cwtmatr,
        'regions'   : np.array(inc_regions),
        'days'      : np.array(onsets)//24
    }
    return return_data
    
ch2r_files = ['3011606.csv', '4251604.csv', '5121612.csv',
              '5221603.csv', 'Channel1_122616.csv', 'Channel2_122616.csv']
control_files = ['Channel9_122616.csv', '2261601.csv', '11191501.csv', 
                     '2231603.csv']
ch2r_paths = 'data/real_data/stim/'
control_paths = 'data/real_data/controls/'
    


# define plotting function
def analyze_experiment(path, filename):
    """ plots the result for a single experiment """
    
    data = load_invivo_data(path+filename, binning = 15)
    
    onset_results = cwt_onset(data['times'], data['counts'])

    
    fig = plt.figure()
    plo.PlotOptions(ticks='in')
    ax = plt.subplot()
    
    #plot stimulations
    ax.add_patch(patches.Rectangle(
        (data['L'][0], data['L'][2]),   # (x,y)
        data['L'][1]-data['L'][0],          # width
        1.0,          # height
        color = 'h', label='Lo'
        ))
    ax.add_patch(patches.Rectangle(
        (data['H'][0], data['H'][2]),   # (x,y)
        data['H'][1]-data['H'][0],          # width
        1.0,          # height
        color = 'j', label='Hi/Lo'
        ))

    
    ax.plot( onset_results['days'], 
            onset_results['onsets']%24, 
    'kx', mew=1, label=filename[:-4])
    #ax.set_ylim([0,24])    
    #plo.format_2pi_axis(ax, y=True, x=False)
    ax.set_xlabel('Day')
    ax.set_ylabel('Activity Onset (h)')
    plt.legend()
    plt.tight_layout()
    
    fig.savefig(path+filename[:-4]+'.png')
    plt.clf()
    plt.close(fig)




for filename in ch2r_files:
    analyze_experiment(ch2r_paths, filename)

for filename in control_files:
    analyze_experiment(control_paths, filename)








