"""
j.h.abel 25 Jan 2017


This includes the plot for Fig 6c.

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


def analyze_full_experiment(path, filename):
    """ get the results for the full experiment. Note: 3 onsets before the stim
    are included!!!!"""

    # load the data, calc the onset times
    data = load_invivo_data(path+filename, binning = 15)
    onset_results = cwt_onset(data['times'], data['counts'])
    onsets = onset_results['onsets']

    print "NOT re-aligning so that every day is relative to start of stim"
    print "do not use for plots of full experiment"
    # calculate the times of the START each stimulation
    stim_times_h = np.linspace(data['H'][0]*24+data['H'][2],
                             data['H'][1]*24+data['H'][2],
                             int(data['H'][1]-data['H'][0]+1), endpoint=True)
    stim_times_l = np.linspace(data['L'][0]*24+data['L'][2],
                             data['L'][1]*24+data['L'][2],
                             int(data['L'][1]-data['L'][0]+1), endpoint=True)

    # for the high freq region
    onset_earliest = np.max(np.where(onsets<np.min(stim_times_h))[0])-2
    onset_latest = np.min(np.where(onsets>np.max(stim_times_h))[0])+1
    onsets_h = onsets[onset_earliest:onset_latest]%24 - stim_times_h[0]%24
    days_h = onset_results['days'][onset_earliest:onset_latest]
    days_h = days_h-np.min(days_h)

    # for the low freq region
    onset_earliest = np.max(np.where(onsets<np.min(stim_times_l))[0])-2
    onset_latest = np.min(np.where(onsets>np.max(stim_times_l))[0])+1
    onsets_l = onsets[onset_earliest:onset_latest]%24 - stim_times_l[0]%24
    days_l = onset_results['days'][onset_earliest:onset_latest]
    days_l = days_l-np.min(days_l)

    return {'days_h': days_h,
            'onsets_h' : onsets_h,
            'onsets_l' : onsets_l,
            'days_l': days_l}


# now that all is set up, analyze the data
chr2_data = [analyze_full_experiment(chr2_paths, filename) for
                    filename in chr2_files]
control_data = [analyze_full_experiment(control_paths, filename) for
                    filename in control_files]




# Sync Index for Each Day
hif_sync = []
lif_sync = []
control_sync = []

hif_ang = []
lif_ang = []
control_ang = []


for day in range(7):
    hif_phases = []
    lif_phases = []
    con_phases = []
    for expt in chr2_data:
        lif_phases.append(expt['onsets_l'][day]*2*np.pi/24)
        hif_phases.append(expt['onsets_h'][day]*2*np.pi/24)
    for expt in control_data:
        con_phases.append(expt['onsets_l'][day]*2*np.pi/24)
        con_phases.append(expt['onsets_h'][day]*2*np.pi/24)
    hif_sync.append(1/len(np.hstack(hif_phases))*\
            np.abs(np.sum(np.exp(1j*np.hstack(hif_phases)))))
    lif_sync.append(1/len(np.hstack(lif_phases))*\
            np.abs(np.sum(np.exp(1j*np.hstack(lif_phases)))))
    control_sync.append(1/len(np.hstack(con_phases))*\
            np.abs(np.sum(np.exp(1j*np.hstack(con_phases)))))
    hif_ang.append(np.angle(1/len(np.hstack(hif_phases))*\
            np.sum(np.exp(1j*np.hstack(hif_phases)))))
    lif_ang.append(np.angle(1/len(np.hstack(lif_phases))*\
            np.sum(np.exp(1j*np.hstack(lif_phases)))))
    control_ang.append(np.angle(1/len(np.hstack(lif_phases))*
                    np.sum(np.exp(1j*np.hstack(con_phases)))))

plo.PlotOptions(ticks='in')

plt.figure(figsize=(1.55,1.65))
gs = gridspec.GridSpec(1,1)

ax=plt.subplot(gs[0,0])
ax.plot(np.arange(7)-2, control_sync, label="Control HIF+LIF")
ax.plot(np.arange(7)-2, hif_sync, label="Control HIF+LIF")
ax.plot(np.arange(7)-2, lif_sync, label="Control HIF+LIF")
ax.set_xlim([-2,4])
ax.set_ylim([0.5,1.0])
ax.set_xticklabels([])
ax.set_ylabel(r'Sync Index')

bx = ax.twinx()
bx.plot(np.arange(7)-2, np.array(control_ang)*12/pi, ls=":", label="Control HIF+LIF")
bx.plot(np.arange(7)-2, np.array(hif_ang)*12/pi, ls=":", label="HIF")
bx.plot(np.arange(7)-2, np.array(lif_ang)*12/pi, ls=":", label="LIF")
bx.set_ylabel("Angle of Entrainment (h)")
bx.set_ylim([-6,2])
#bx=plt.subplot(gs[1,0])
#bx.plot(np.arange(7)-2, control_ang, label="Control HIF+LIF")
#bx.plot(np.arange(7)-2, hif_ang, label="HIF")
#bx.plot(np.arange(7)-2, lif_ang, label="LIF")
#bx.set_xlim([-2,4])
ax.set_xlabel('Day Relative\nto Stim')
#bx.set_ylabel(r'Entrained Angle')
#plt.legend()
plt.tight_layout(**plo.layout_pad)



# NOW, THE BOXPLOT



# break the data into hi, lo, none stim regions
def breakdown_experiment(path, filename):
    """ plots the result for a single experiment """

    # load the data, calc the onset times
    data = load_invivo_data(path+filename, binning = 15)
    onset_results = cwt_onset(data['times'], data['counts'])
    onsets = onset_results['onsets']

    # calculate the CENTER times of each stimulation
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

    return {'h_angles'  : h_angles+24,
            'h_changes' : h_changes+24,
    		'l_angles'  : l_angles+24,
    		'l_changes' : l_changes+24,
    		'off_diffs' : off_diffs+24}







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


plo.PlotOptions(ticks='in')

# distribution of shifts
boxplot_data0 = chr2_results['h_changes']
boxplot_data1 = chr2_results['l_changes']
boxplot_data2 = chr2_results['off_diffs']
boxplot_data3 = control_results['h_changes']
boxplot_data4 = control_results['l_changes']
boxplot_data5 = control_results['off_diffs']


boxplot_data = [boxplot_data5, boxplot_data3, boxplot_data4, boxplot_data2,
                boxplot_data0, boxplot_data1]

names = ['Control\nNo Stim','Control\nHIF','Control\nLIF', 'VIPChR2\nNo Stim',
         'VIPChR2\nHIF','VIPChR2\nLIF']
plt.figure(figsize = (3.5,1.8))
ax = plt.subplot()

means = [np.mean(data) for data in boxplot_data]
sems = [stats.sem(data) for data in boxplot_data]
colors = ['gray']*4+['fl']*2
N = 6
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

rects = ax.bar(ind[:4], means[:4], width, color='gray', yerr=sems[:4])
rects1 = ax.bar(ind[4:], means[4:], width, color='fl', yerr=sems[4:])

ax.set_xlabel('Condition')
ax.set_ylabel('Mean Period Length (h)')
ax.set_ylim([22,25.5])
ax.set_xticks([0,1,2,3,4,5])
ax.set_xticklabels(names)
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















#
#
#
#
# RAYLEIGH NO LONGER USED

def rayleigh_plot(ax, analyzed_data, rad=1, color='f', refine=False):
    """ rayleigh plot for this specific setup"""
    marks = ['o','s','d','^','h','P','*']
    for i, data in enumerate(analyzed_data):
        ax.plot(-data*2*np.pi/24+np.pi/2, rad, color=color, marker = 'o',
                markersize=3, alpha = (0.2+0.75*(i+1)/(len(analyzed_data))),
                zorder=-20)

    if refine==True:
        ax.set_rticks([])
        ax.set_rmax(rad+0.1)
        ax.set_rmin(0)
        ax.set_xticklabels([-6,-3,0,3,6,9,12,-9])
        ax.grid(True, color='gray', ls=':', lw=0.5)

plo.PlotOptions(ticks='in')
palatte = colorb.get_map('Dark2','Qualitative',7)
colors = palatte.mpl_colors[1:]
print "Note: excluding the 2-days of stim chr2 mouse and control mouse."

plt.figure(figsize=(3.5,1.6))
gs = gridspec.GridSpec(1,3)
ax = plt.subplot(gs[0,0],projection='polar')

hif_phases = []
for i,expt in enumerate(chr2_data):
    rayleigh_plot(ax, expt['onsets_h'][3:7], rad = 1+0.1*i,
                  color=colors[i], refine=np.floor(i/len(expt)))
    hif_phases.append(expt['onsets_h'][3:7]*2*np.pi/24)

hif_rad = 1/len(np.hstack(hif_phases))*\
            np.abs(np.sum(np.exp(1j*np.hstack(hif_phases))))
hif_angle = np.mean(np.hstack(hif_phases))
ax.plot([-hif_angle+np.pi/2,-hif_angle+np.pi/2],[1E-11,hif_rad*(1+0.1*i)],'k')
ax.set_xlabel('VIPChR2 HIF\nSI = '+str(np.round(hif_rad,2)))
ax.set_rasterization_zorder(-10)

bx = plt.subplot(gs[0,1],projection='polar')
lif_phases = []
for i,expt in enumerate(chr2_data):
    rayleigh_plot(bx, expt['onsets_l'][3:7], rad = 1+0.1*i,
                  color=colors[i], refine=np.floor(i/len(expt)))

    lif_phases.append(expt['onsets_l'][3:7]*2*np.pi/24)

lif_rad = 1/len(np.hstack(lif_phases))*\
            np.abs(np.sum(np.exp(1j*np.hstack(lif_phases))))
lif_angle = np.mean(np.hstack(lif_phases))
bx.plot([-lif_angle+np.pi/2,-lif_angle+np.pi/2],[1E-11,lif_rad*(1+0.1*i)],'k')
bx.set_xlabel('VIPChR2 LIF\nSI = '+str(np.round(lif_rad,2)))
bx.set_rasterization_zorder(-10)

cx = plt.subplot(gs[0,2],projection='polar')
con_phases_l = []
con_phases_h = []
for i,expt in enumerate(control_data):
    rayleigh_plot(cx, expt['onsets_l'][3:7], rad = 1+0.1*i,
                  color=colors[i], refine=np.floor(i/len(expt)))
    rayleigh_plot(cx, expt['onsets_h'][3:7], rad = 1+0.1*i,
                  color=colors[i], refine=True)
    con_phases_l.append(expt['onsets_l'][3:7]*2*np.pi/24)
    con_phases_h.append(expt['onsets_h'][3:7]*2*np.pi/24)

conh_rad = 1/len(np.hstack(con_phases_h))*\
            np.abs(np.sum(np.exp(1j*np.hstack(con_phases_h))))
conh_angle = np.mean(np.hstack(con_phases_h))
conl_rad = 1/len(np.hstack(con_phases_l))*\
            np.abs(np.sum(np.exp(1j*np.hstack(con_phases_l))))
conl_angle = np.mean(np.hstack(con_phases_l))
cx.plot([-con_angle+np.pi/2,-con_angle+np.pi/2],[1E-11,con_rad*(1+0.1*i)],'k')
cx.set_xlabel('Control HIF+LIF\nSI = '+str(np.round(con_rad,2)))
cx.set_rasterization_zorder(-10)

plt.tight_layout(**plo.layout_pad)
print hif_rad, lif_rad, conh_rad, conl_rad

plt.savefig("data/rayleigh_example.svg", dpi=900)




























