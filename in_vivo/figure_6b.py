# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 17:42:11 2017

@author: abel

Plots the whole experiment at once
"""

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
import brewer2mpl as colorb
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from LocalImports import PlotOptions as plo
from LocalImports import Bioluminescence as bl


import pywt
print "pywt version should be 0.5.2, it is "+str(pywt.__version__)

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



chr2_files = ['4251604.csv', '5121612.csv',
              '5221603.csv', 'Channel1_122616.csv', 'Channel2_122616.csv']
control_files = ['Channel9_122616.csv', '2261601.csv', '11191501.csv',
                     ]
chr2_paths = 'data/rayleigh_data/stim/'
control_paths = 'data/rayleigh_data/controls/'


def analyze_full_experiment(path, filename):
    """ plots the result for a single experiment """

    # load the data, calc the onset times
    data = load_invivo_data(path+filename, binning = 15)
    onset_results = cwt_onset(data['times'], data['counts'])
    onsets = onset_results['onsets']

    # calculate the times of each stimulation
    stim_times_h = np.linspace(data['H'][0]*24+data['H'][2],
                             data['H'][1]*24+data['H'][2],
                             int(data['H'][1]-data['H'][0]+1), endpoint=True)
    stim_times_l = np.linspace(data['L'][0]*24+data['L'][2],
                             data['L'][1]*24+data['L'][2],
                             int(data['L'][1]-data['L'][0]+1), endpoint=True)

    # for the high freq region
    # find if first stim is same day as last prestim onset
    print "NOTE: shift values selected for ALIGNING THESE PLOTS, do not use"+\
            " for Rayleigh statistics for each day!!!"
    if stim_times_h[0]-onsets[np.max(np.where(onsets<np.min(stim_times_h))[0])]<12:
        shift_val = 2
    else:
        shift_val = 1
    print shift_val
    onset_earliest = np.max(np.where(onsets<np.min(stim_times_h))[0])-shift_val
    onset_latest = np.min(np.where(onsets>np.max(stim_times_h))[0])+1
    onsets_h = onsets[onset_earliest:onset_latest]%24 - stim_times_h[0]%24
    days_h = onset_results['days'][onset_earliest:onset_latest]
    days_h = days_h-np.min(days_h)

    # for the low freq region
    # find if first stim is same day as last prestim onset
    if stim_times_l[0]-onsets[np.max(np.where(onsets<np.min(stim_times_l))[0])]<12:
        shift_val = 2
    else:
        shift_val = 1
    print shift_val
    onset_earliest = np.max(np.where(onsets<np.min(stim_times_l))[0])-shift_val
    onset_latest = np.min(np.where(onsets>np.max(stim_times_l))[0])+1
    onsets_l = onsets[onset_earliest:onset_latest]%24 - stim_times_l[0]%24
    days_l = onset_results['days'][onset_earliest:onset_latest]
    days_l = days_l-np.min(days_l)

    return {'days_h': days_h,
            'onsets_h' : onsets_h,
            'onsets_l' : onsets_l,
            'days_l': days_l}


def plot_experiments(ax, path, expts, stim='h',
                     colors=None, markers=None):
    """ plots all expts of the same type """
    if markers is None:
        markers = ['o']*len(expts)
    if colors is None:
        colors=['k']*len(expts)
    for idx,expt in enumerate(expts):
        analyzed = analyze_full_experiment(path, expt)
        ax.plot(analyzed['onsets_'+stim], analyzed['days_'+stim]-2,
                 marker=markers[idx%len(markers)], color=colors[idx])
    ax.set_xlim([-12,12])
    ax.set_ylim([-2,7])
    ax.set_yticks([-2,0,2,4,6])
    ax.set_yticklabels([0,2,4,6,8])
    ax.add_patch(patches.Rectangle(
            (0,0), 1, 10, color='gray', alpha=0.4))
    plt.gca().invert_yaxis()


palattelif = colorb.get_map('Set1','Qualitative',7)
palattehif = colorb.get_map('Dark2','Qualitative',7)
ccolors = palattelif.mpl_colors[1:]
ecolors = palattehif.mpl_colors[1:]



plo.PlotOptions(ticks='in')
plt.figure(figsize=(3.5,1.7))
gs = gridspec.GridSpec(1,3)

ax = plt.subplot(gs[0,0])
plot_experiments(ax, control_paths, control_files, colors=ccolors, markers=['s'])
plot_experiments(ax, control_paths, control_files, colors=ccolors, stim='l')

bx = plt.subplot(gs[0,1])
plot_experiments(bx, chr2_paths, chr2_files, colors=ecolors, markers=['s'])
bx.set_yticklabels([])
dx = plt.subplot(gs[0,2])
plot_experiments(dx, chr2_paths, chr2_files, colors=ecolors, stim='l')
dx.set_yticklabels([])


ax.set_title('Control HIF+LIF')
bx.set_title('HIF')
dx.set_title('LIF')
bx.set_xlabel('Relative Activity Onset (h)')

plt.tight_layout(**plo.layout_pad)








