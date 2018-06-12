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
chr2_files = [  '4251604.csv', '5121612.csv',
              '5221603.csv', 'Channel1_122616.csv', 'Channel2_122616.csv']
control_files = ['Channel9_122616.csv', '2261601.csv', '11191501.csv', 
                     ]
chr2_paths = 'data/rayleigh_data/stim/'
control_paths = 'data/rayleigh_data/controls/'
    


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
                             data['H'][1]-data['H'][0]+1, endpoint=True)+0.5
    stim_times_l = np.linspace(data['L'][0]*24+data['L'][2],
                             data['L'][1]*24+data['L'][2],
                             data['L'][1]-data['L'][0]+1, endpoint=True)+0.5
    
    # get the onsets before and after each stim
    # get the before angle
    # get how that angle changes
    h_changes = [] # how the phase angle moves
    h_angles  = [] # what the phase angle is
    for stim_time in stim_times_h:
    	# get onset before
    	before_onset = np.max(onsets[np.where(onsets<stim_time)])
    	# get onset after
    	after_onset =  np.min(onsets[np.where(onsets>stim_time)])
    	# get change, don't ignore it, but angle should not be 0
    	h_change = (after_onset%24-before_onset%24)/((after_onset-before_onset)/24)
    	h_angle  = (before_onset%24-stim_time%24)
    	if np.abs(h_angle) > 0:
    		h_angles.append(h_angle)
    		h_changes.append(h_change)
    h_changes = (np.asarray(h_changes)+12)%24-12
    h_angles  = (np.asarray(h_angles)+12)%24-12
    
    # get the before angle
    # get how that angle changes
    l_changes = [] # how the phase angle moves
    l_angles  = [] # what the phase angle is
    for stim_time in stim_times_l:
    	before_onset = np.max(onsets[np.where(onsets<stim_time)])
    	after_onset =  np.min(onsets[np.where(onsets>stim_time)])
    	l_change = (after_onset%24-before_onset%24)
    	l_angle  = (before_onset%24-stim_time%24)
    	if np.abs(l_angle) > 0:
    		l_angles.append(l_angle)
    		l_changes.append(l_change)
    l_changes = (np.asarray(l_changes)+12)%24-12
    l_angles  = (np.asarray(l_angles)+12)%24-12
    
    # find out where the onsets are for stimulation off
    all_stim = np.hstack([stim_times_l, stim_times_h])
    off_diffs = []
    for idx,onset in enumerate(onsets[:-1]):
    	next_onset = onsets[idx+1]
        if len(np.where(np.logical_and(all_stim>onset, 
        				all_stim<next_onset))[0]) < 1:
        	# if there is no stim between this and the next offset, add it to the list
        	off_diffs.append(next_onset%24-onset%24)
    off_diffs = (np.asarray(off_diffs)+12)%24-12
    
    return {'h_angles'  : h_angles,
            'h_changes' : h_changes,
    		'l_angles'  : l_angles,
    		'l_changes' : l_changes,
    		'off_diffs' : off_diffs}
    

    




chr2_results = {'h_angles'  : [],
                'h_changes' : [],
    		     'l_angles'  : [],
    		     'l_changes' : [],
    		     'off_diffs' : []}
for filename in chr2_files:
    result = breakdown_experiment(chr2_paths, filename)
    for key in chr2_results.keys():
        chr2_results[key] = np.hstack([chr2_results[key], result[key]])



control_results = {'h_angles'  : [],
            'h_changes' : [],
    		'l_angles'  : [],
    		'l_changes' : [],
    		'off_diffs' : []}
for filename in control_files:
    result = breakdown_experiment(control_paths, filename)
    for key in control_results.keys():
        control_results[key] = np.hstack([control_results[key], result[key]])


plo.PlotOptions()

# distribution of shifts
boxplot_data0 = chr2_results['h_changes']
boxplot_data1 = chr2_results['l_changes']
boxplot_data2 = chr2_results['off_diffs']
boxplot_data3 = control_results['h_changes']
boxplot_data4 = control_results['l_changes']
boxplot_data5 = control_results['off_diffs']


boxplot_data = [boxplot_data5, boxplot_data3, boxplot_data4, boxplot_data2,
                boxplot_data0, boxplot_data1]

names = ['Control\nOff','Control\nHIF','Control\nLIF', 'VIPchr2\nOff',
         'VIPchr2\nHIF','VIPchr2\nLIF']
plt.figure(figsize = (3.5,1.8))
ax = plt.subplot()
ax.axhline(0,xmin=0,xmax=6,ls='--',color='k')
box = ax.boxplot(boxplot_data, sym='.', patch_artist=True)
colors = ['gray']*4+['il']*2
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels(names)
#ax.text(0.6, 0, '* p<1E-4, FWER<0.05',
#        horizontalalignment='left',
#        verticalalignment='bottom',
#        transform=ax.transAxes)
ax.set_xlabel('Condition')
ax.set_ylabel('Daily Change in\nActivity Onset (h)')
ax.set_ylim([-6.5,6.5])
ax.set
plt.tight_layout(**plo.layout_pad)


# statistics for the box plot
stats.f_oneway(boxplot_data0,boxplot_data1,boxplot_data2, boxplot_data3, 
				boxplot_data4,boxplot_data5)


boxplot_data = [boxplot_data5,boxplot_data3,boxplot_data4, boxplot_data2, 
				boxplot_data0,boxplot_data1]
rec_array = []
for idx,condition in enumerate(['co','ch','cl','o','h','l']):
    for slope in boxplot_data[idx]:
        rec_array.append((slope,condition))
    
rec_array = np.rec.array(rec_array, dtype=[('slope','float64'),('condition','|S8')])
tukeysync = pairwise_tukeyhsd(rec_array['slope'],rec_array['condition'])
print tukeysync



# plotting the data vertically like how cristina did































# is it entraining? two scatter plots to test
plt.figure()
plt.plot(control_results['l_angles'], control_results['l_changes'], 'ko', 
         label='Control')
plt.plot(chr2_results['l_angles'], chr2_results['l_changes'], 'ro', 
         label='chr2')
plt.axhline(0, xmin=-12,xmax=12,ls='--',color='k')
plt.axvline(0, ymin=-12,ymax=12,ls='--',color='k')
plt.title('Low Frequency')
plt.xlabel('Activity Onset Relative to Stimulus (h)')
plt.ylabel('Daily Change in Onset Relative to Stimulus (h)')
plt.tight_layout(**plo.layout_pad)
print 'chr2 low'
print stats.linregress(chr2_results['l_angles'], chr2_results['l_changes'])
print 'control low'
print stats.linregress(control_results['l_angles'], control_results['l_changes'])

plt.figure()
plt.plot(control_results['h_angles'], control_results['h_changes'], 'ko',
         label='Control')
plt.plot(chr2_results['h_angles'], chr2_results['h_changes'], 'fo',
         label='chr2')
plt.axhline(0, xmin=-12,xmax=12,ls='--',color='k')
plt.axvline(0, ymin=-12,ymax=12,ls='--',color='k')
plt.title('High Frequency')
plt.xlabel('Activity Onset Relative to Stimulus (h)')
plt.ylabel('Daily Change in Onset Relative to Stimulus (h)')
plt.tight_layout(**plo.layout_pad)
print 'chr2 hi'
print stats.linregress(chr2_results['h_angles'], chr2_results['h_changes'])
print 'control hi'
print stats.linregress(control_results['h_angles'], control_results['h_changes'])





