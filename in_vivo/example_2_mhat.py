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
from matplotlib import gridspec

import PlotOptions as plo
import Bioluminescence as bl


import pywt 

example_path1 = 'data/example2/Channel1.csv'
example_path2 = 'data/example2/Channel2.csv'
example_path3 = 'data/example2/Channel9.csv'

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
    widths = np.arange((60/bin_param), 
                       10*(60/bin_param), 
                       0.05*(60/bin_param))
    
    #pywt version
    cwtmatr, freqs = pywt.cwt(y, widths, 'mexh')
    periods = 1/freqs/(60/bin_param)
    
    inst_per_loc = np.argmax(np.abs(cwtmatr.T),1)
    inst_ampl = np.asarray([cwtmatr.T[idx, loc] 
                            for idx,loc in enumerate(inst_per_loc)])
    

    # identify regions of increasing activity
    maxs = signal.argrelextrema(inst_ampl, np.greater, order=20)[0]
    mins = signal.argrelextrema(inst_ampl, np.less, order=20)[0]
    
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
        onsets.append(find_cwt0_region(region,x,inst_ampl))
    
    
    return_data = {
        'x'         : x,
        'y'         : y,
        'onsets'    : np.array(onsets),
        'dwt'       : cwtmatr,
        'regions'   : np.array(inc_regions),
        'days'      : np.array(onsets)//24
    }
    return return_data
    
example_paths = [example_path1, example_path2, example_path3]

all_cwts = []
for path in example_paths:
    times, counts = load_invivo_data(path, binning = 15)
    all_cwts.append(cwt_onset(times, counts))
    








names = ['Channel 1', 'Channel 2', 'Channel 9']


plt.figure()
plo.PlotOptions(ticks='in')
ax = plt.subplot()

ax.plot([4,11], np.array([15,15]), color='f', lw=2, label='Lo')
ax.plot([15,28], np.array([10,10]), color='f', ls='--', lw=2, label='Hi/Lo')

ax.plot( all_cwts[0]['days'], all_cwts[0]['onsets']%24, 
'fx', mew=1, label='Channel1 (VIPCre/+;Ai32/+)')
ax.plot( all_cwts[1]['days'], all_cwts[1]['onsets']%24, 
'f+', mew=1, label='Channel2 (VIPCre/+;Ai32/+)')
ax.plot( all_cwts[2]['days'], 
        np.unwrap(all_cwts[2]['onsets']%24 *2*np.pi/24)*24/(2*np.pi), 
        'k.', mew=1, label='Channel9 (Control)')
ax.set_xlim([-1,28])



#plo.format_2pi_axis(ax, y=True, x=False)
ax.set_xlabel('Day')
ax.set_ylabel('Activity Onset (h)')
plt.legend()
plt.tight_layout()




plt.plot(pulse_timing2[:7]/24, pulse_2_phases)












for i,wav in enumerate(wavs):
    plt.plot(wav['x']/24., wav['period'], label = names[i])
plt.legend()
plt.ylim([22,25])



# actogram test
import timbre.readdata as rd


cols = {'day':'Day', 'hour':'Hour', 'min':'Min', 'val':'Cnt/min'}

def my_coltypes(line):
    # line = {'date': '2010-01-01', 'time': '10:00:00', 'val': '50'}
    day = line['day']
    hour = line['hour']
    min_ = line['min']
    # convert values from string to int
    year, mon, day = int(0), int(0), int(day)
    hour, min_, sec = int(hour), int(min_), int(0)

    result = {}
    result['time'] = datetime.datetime(2016, 12, day, hour, min_, 0)
    result['cnt'] = int(line['val'])
    result['cnt'] = int(result['cnt'])
    return result


dataset = rd.CSVReader(example_path1, columns=cols, coltypes=my_coltypes, 
                       delimiter=',', skiplines=1)

actogram(dataset, 'cnt')












