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


def cwt_phases(times, counts):
    """ takes the phase at the start of each day from the morse CWT """
    # get the wavelet result
    wavres = bl.continuous_wavelet_transform(times, counts, 
                                         shortestperiod=10, longestperiod = 40)
    # figure out where each day starts
    locs = []
    timing = np.arange(1,int(np.max(times/24)))*24
    for time in timing:
        locs.append(np.argmin((wavres['x']-time)**2))
    # get the phase there
    ret_pulse_phis = wavres['phase'][locs]
    ret_pulse_amps = np.array([np.max(wavres['cwt_relamp'][:,loc])
                        for loc in locs])
    # only keep where phase is identifiable
    phis = np.unwrap(ret_pulse_phis[np.where(ret_pulse_amps>0.10)])
    phitimes = np.array(timing)[np.where(ret_pulse_amps>0.10)]
    
    return_data = {
        'x'         : times,
        'y'         : counts,
        'phis'      : phis,
        'days'      : np.array(phitimes)//24,
        'wav'       : wavres
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
    
    data = load_invivo_data(path+filename, binning = 8)
    
    phase_results = cwt_phases(data['times'], data['counts'])

    
    fig = plt.figure()
    plo.PlotOptions(ticks='in')
    ax = plt.subplot()
    
    #plot stimulations
    
    ax.add_patch(patches.Rectangle(
        (data['L'][0], -data['L'][2]*2*np.pi/24+2*np.pi-2),   # (x,y)
        data['L'][1]-data['L'][0],          # width
        1.0*2*np.pi/24,          # height
        color = 'h', label='Lo'
        ))
    ax.add_patch(patches.Rectangle(
        (data['H'][0], -data['H'][2]*2*np.pi/24+2*np.pi-2),   # (x,y)
        data['H'][1]-data['H'][0],          # width
        1.0*2*np.pi/24,          # height
        color = 'j', label='Hi/Lo'
        ))
    

    
    ax.plot( phase_results['days'], 
            phase_results['phis'], 
    'kx', mew=1, label=filename[:-4])
    #ax.set_ylim([0,24])    
    plo.format_2pi_axis(ax, y0=True, x=False)
    ax.set_xlabel('Day')
    ax.set_ylabel('Running Phase (rad)')
    plt.legend()
    plt.tight_layout()
    
    fig.savefig(path+filename[:-4]+'phases.png')
    plt.clf()
    plt.close(fig)




for filename in ch2r_files:
    analyze_experiment(ch2r_paths, filename)

for filename in control_files:
    analyze_experiment(control_paths, filename)








