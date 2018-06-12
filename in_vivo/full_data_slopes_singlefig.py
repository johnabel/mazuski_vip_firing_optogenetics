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
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from LocalImports import PlotOptions as plo
from LocalImports import Bioluminescence as bl


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
#
ch2r_files = [ '3011606.csv', '4251604.csv', '5121612.csv',
              '5221603.csv', 'Channel1_122616.csv', 'Channel2_122616.csv']
control_files = ['Channel9_122616.csv', '2261601.csv', '11191501.csv', 
                     '2231603.csv']
ch2r_paths = 'data/real_data/stim/'
control_paths = 'data/real_data/controls/'

def fit_slope(times, onsets):
    """ returns linear regression of points if the length of this region is 
    longer than 3 """
    if len(times) >= 3:
        return stats.linregress(times,onsets)[0]*24.

# break the data into hi, lo, none stim regions
def breakdown_experiment(path, filename):
    """ plots the result for a single experiment """
    
    # load the data, calc the onset times
    data = load_invivo_data(path+filename, binning = 15)
    onset_results = cwt_onset(data['times'], data['counts'])
    onsets = onset_results['onsets']
    
    # calculate the times of each stimulation
    stim_times_h = np.linspace(data['H'][0]*24+data['H'][2],
                             data['H'][1]*24+data['H'][2],
                             int(data['H'][1]-data['H'][0]+0.5),
                             endpoint=True)+0.5
    stim_times_l = np.linspace(data['L'][0]*24+data['L'][2],
                             data['L'][1]*24+data['L'][2],
                             int(data['L'][1]-data['L'][0]+0.5),
                             endpoint=True)+0.5
    
    stim_times_all = np.sort(np.hstack([stim_times_h,stim_times_l]))
    #all times in order
    
    # off slopes
    o_slope = []
    o_dur =[]
    
    first_region_times = onsets[np.where(onsets<np.min(stim_times_all))[0]]
    first_region_onsets = np.unwrap((first_region_times%24)*np.pi/12)*12/np.pi
    if len(first_region_times) >= 3:
        o_slope.append(fit_slope(first_region_times, first_region_onsets))
        o_dur.append((np.max(first_region_times)-np.min(first_region_times))/24)
    
    mid_start = np.min([np.max(stim_times_l), np.max(stim_times_h)])
    mid_end   = np.max([np.min(stim_times_l), np.min(stim_times_h)])
    middle_times = onsets[
                np.where(np.logical_and(onsets>mid_start, onsets<mid_end))[0]]
    middle_onsets = np.unwrap((middle_times%24)*np.pi/12)*12/np.pi
    if len(middle_times) >= 3:
        o_slope.append(fit_slope(middle_times, middle_onsets))
        o_dur.append((np.max(middle_times)-np.min(middle_times))/24)
    
    end_times = onsets[np.where(onsets>np.max(stim_times_all))[0]]   
    end_onsets = np.unwrap((end_times%24)*np.pi/12)*12/np.pi
    if len(end_times) >= 3:
        o_slope.append(fit_slope(end_times, end_onsets))
        o_dur.append((np.max(end_times)-np.min(end_times))/24)
    
    
    lo_start = np.max(np.where(onsets<np.min(stim_times_l))[0])
    lo_end = np.min(np.where(onsets>np.max(stim_times_l))[0])
    lo_times = onsets[lo_start:lo_end+1]
    lo_onsets = np.unwrap((lo_times%24)*np.pi/12)*12/np.pi
    l_slope = [fit_slope(lo_times, lo_onsets)]
    l_dur = [(onsets[lo_start]-onsets[lo_end])/24]
    l_angle  = [(lo_onsets[0]%24-stim_times_l[0]+12)%24-12]
    
    hi_start = np.max(np.where(onsets<np.min(stim_times_h))[0])
    hi_end = np.min(np.where(onsets>np.max(stim_times_h))[0])
    hi_times = onsets[hi_start:hi_end+1]
    hi_onsets = np.unwrap((hi_times%24)*np.pi/12)*12/np.pi
    h_slope = [fit_slope(hi_times, hi_onsets)]
    h_dur = [(onsets[hi_start]-onsets[hi_end])/24]
    h_angle  = [(hi_onsets[0]%24-stim_times_h[0]+12)%24-12]

    
    return {'h_slope' : h_slope,
            'h_angle' : h_angle,
            'h_dur'   : h_dur,
            'l_slope' : l_slope,
            'l_angle' : l_angle,
            'l_dur'   : l_dur,
    		 'o_slope' : o_slope,
            'o_dur'   : o_dur
       }
    

    



ch2r_results = {'h_slope' : [],
            'h_angle' : [],
            'h_dur'   : [],
            'l_slope' : [],
            'l_angle' : [],
            'l_dur'   : [],
    		 'o_slope' : [],
            'o_dur'   : []}

for filename in ch2r_files:
    result = breakdown_experiment(ch2r_paths, filename)
    for key in ch2r_results.keys():
        ch2r_results[key] = np.hstack([ch2r_results[key], result[key]])



control_results = {'h_slope' : [],
                'l_slope' : [],
                'o_slope' : []}
                
for filename in control_files:
    result = breakdown_experiment(control_paths, filename)
    for key in control_results.keys():
        control_results[key] = np.hstack([control_results[key], result[key]])



plo.PlotOptions()



















