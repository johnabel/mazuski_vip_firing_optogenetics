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
import brewer2mpl as colorb

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
        digitized = np.digitize(ret_times, bins)
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
    zeros_loc = np.where(np.diff(np.sign(specific_region)))[0]
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
chr2_files = [  '4251604.csv', '5121612.csv',
              '5221603.csv', 'Channel1_122616.csv', 'Channel2_122616.csv',
                'longmouse1.csv', 'longmouse2.csv']
control_files = ['Channel9_122616.csv', '2261601.csv', '11191501.csv',
                     ]
chr2_paths = 'data/prc_data/stim/'
control_paths = 'data/prc_data/controls/'


# break the data into hi, lo, none stim regions
def experiment_phase_response(path, filename):
    """ looks at phase response for center of stimulus """

    # load the data, calc the onset times
    data = load_invivo_data(path+filename, binning = 15)
    onset_results = cwt_onset(data['times'], data['counts'])
    onsets = onset_results['onsets']

    # calculate the times of each stimulation
    stim_times_h = np.linspace(data['H'][0]*24+data['H'][2],
                             data['H'][1]*24+data['H'][2],
                             data['H'][1]-data['H'][0]+1, endpoint=True)+0.5
    stim_times_l = np.linspace(data['L'][0]*24+data['L'][2],
                             data['L'][1]*24+data['L'][2],
                             data['L'][1]-data['L'][0]+1, endpoint=True)+0.5

    h_timeofstim = [] #
    h_perlen  = [] #
    if stim_times_h[0] > 1: # if not, this is not used
        for stim_time in stim_times_h:
            if stim_time<np.max(onsets):
                # get onset before
                before_onset = np.max(onsets[np.where(onsets<stim_time)])
                # get onset after
                after_onset =  np.min(onsets[np.where(onsets>stim_time)])
                # get change, don't ignore it, but angle should not be 0
                per_len_h = after_onset-before_onset
                time_of_stim_relto_onset = after_onset-stim_time
                # NOTE TO SELF: ONSET BEFORE STIM IS DELAY SO NEGATIVE
                if time_of_stim_relto_onset>12:
                    time_of_stim_relto_onset = before_onset-stim_time
                h_perlen.append(per_len_h)
                h_timeofstim.append(time_of_stim_relto_onset)


    l_timeofstim = [] #
    l_perlen  = [] #
    if stim_times_l[0] > 1: # if not, this is not used
        for stim_time in stim_times_l:
            if stim_time<np.max(onsets):
                # get onset before
                before_onset = np.max(onsets[np.where(onsets<stim_time)])
                # get onset after
                after_onset =  np.min(onsets[np.where(onsets>stim_time)])
                # get change, don't ignore it, but angle should not be 0
                per_len_l = after_onset-before_onset
                time_of_stim_relto_onset = after_onset-stim_time
                # NOTE TO SELF: ONSET BEFORE STIM IS DELAY SO NEGATIVE
                if time_of_stim_relto_onset>12:
                    time_of_stim_relto_onset = before_onset-stim_time
                l_perlen.append(per_len_l)
                l_timeofstim.append(time_of_stim_relto_onset)

    all_stim = np.hstack([stim_times_l, stim_times_h])
    off_diffs = []
    for idx,onset in enumerate(onsets[:-1]):
        next_onset = onsets[idx+1]
        # want to ensure we are not in middle of stim with short period, check 2h on either side
        if len(np.where(np.logical_and(all_stim+2>onset,
        	 			all_stim-2<next_onset))[0]) < 1:
            # if there is no stim between this and the next offset, add it to the list
            off_diffs.append(next_onset%24-onset%24)

    off_diffs = (np.asarray(off_diffs)+12)%24-12
    nonstim_period = off_diffs.mean()+24

    '''
    return {'h_pers'  : np.array(h_perlen),
            'h_times' : np.array(h_timeofstim),
    		'l_pers'  : np.array(l_perlen),
    		'l_times' : np.array(l_timeofstim),
            }
    '''
    return {'h_delphi'  : nonstim_period-np.array(h_perlen),
            'h_times' : np.array(h_timeofstim),
    		'l_delphi'  : nonstim_period-np.array(l_perlen),
    		'l_times' : np.array(l_timeofstim)}



chr2_results = {'h_delphi'  : [],
                'h_times' : [],
    		     'l_delphi'  : [],
    		     'l_times' : []}
for filename in chr2_files:
    result = experiment_phase_response(chr2_paths, filename)
    for key in chr2_results.keys():
        chr2_results[key] = np.hstack([chr2_results[key], result[key]])



control_results = {'h_delphi'  : [],
                'h_times' : [],
    		     'l_delphi'  : [],
    		     'l_times' : []}
for filename in control_files:
    result = experiment_phase_response(control_paths, filename)
    for key in control_results.keys():
        control_results[key] = np.hstack([control_results[key], result[key]])

period = 23.24 # period of control mice

palatte = colorb.get_map('Dark2','Qualitative',7)
colors = palatte.mpl_colors


#
#
# Option #2

plo.PlotOptions(ticks='in')

control_phase_shifts =np.hstack([control_results['l_delphi'],control_results['h_delphi']])
control_phases = np.hstack([control_results['l_times'],control_results['h_times']])
control_phases = control_phases[np.where(control_phase_shifts>-5)[0]]
control_phase_shifts = control_phase_shifts[np.where(np.abs(control_phase_shifts)<5)[0]]

chr2_phase_shifts =np.hstack([chr2_results['l_delphi'],chr2_results['h_delphi']])
chr2_phases = np.hstack([chr2_results['l_times'],chr2_results['h_times']])
chr2_phases = chr2_phases[np.where(chr2_phase_shifts>-5)[0]]
chr2_phase_shifts = chr2_phase_shifts[np.where(np.abs(chr2_phase_shifts)<5)[0]]


def fit_spline_running(times,shifts,discretization=48,binsize=5,period=period):
    """ fits a spline to a running mean """
    ts = np.linspace(0,period, discretization+1)
    vals = []
    for time in ts:
        valid1 = np.where(np.logical_and(times<time+binsize/2
                                        , times>time-binsize/2))[0]
        valid2 = np.where(np.logical_and(times-period<time+binsize/2
                                        , times-period>time-binsize/2))[0]
        valid3 = np.where(np.logical_and(times+period<time+binsize/2
                                        , times+period>time-binsize/2))[0]
        meanshift = np.mean(shifts[np.hstack([valid1,valid2,valid3])])
        vals.append(meanshift)
    vals = np.asarray(vals)

    spl = UnivariateSpline(np.hstack([ts-period,ts,ts+period]),
                                     np.hstack([vals,vals,vals]),k=3,s=1)

    return ts,vals,spl

plt.figure(figsize=(1.67,1.05))
ax = plt.subplot()
ax.plot(12-control_phases, control_phase_shifts, ls='', color=colors[0],
        marker='o',
        markersize=1.2, alpha=0.6)
ax.plot(12-chr2_phases, chr2_phase_shifts, ls='', color=colors[1], marker='o',
        markersize=1.2,alpha=0.6)
ax.set_xlabel('CT (h)')
ax.set_ylabel('$\Delta\phi$(h)')
ax.set_xticks([0,4,8,12,16,20])
ax.axhline(0,0,24,color='k',ls='--')
ax.set_ylim([-3.1,2.1])


xs,ys,spline = fit_spline_running(12-control_phases, control_phase_shifts, binsize=5)
ts = np.arange(0,period,0.01)
plt.plot(ts,spline(ts), color=colors[0], lw=1., label='Control')
xs,ys,spline = fit_spline_running(12-chr2_phases, chr2_phase_shifts, binsize=5)
ts = np.arange(0,period,0.01)
plt.plot(ts,spline(ts), color=colors[1], lw=1., label='VIPChR2')
plt.legend()
# 23.24 is the control period
# point of entrainment is where the center of the pulse gives -0.76; 13.93h=CT
#plt.plot(xs,ys)

ax.set_xlim([0,period])
plt.tight_layout(**plo.layout_pad)













