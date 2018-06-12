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
        digitized = np.digitize(ret_times, bins)
        new_counts = np.array([ret_counts[digitized == i].sum() 
                        for i in range(1, len(bins))])
        # re-assign                    
        ret_counts = new_counts
        ret_times = new_times
    
    return ret_times, ret_counts


def actogram(times, counts):
    """ double plotted actogram"""
    pass
    
    



times1, counts1 = load_invivo_data(example_path1, binning = 8)
times2, counts2 = load_invivo_data(example_path2, binning = 8)
times3, counts3 = load_invivo_data(example_path3, binning = 8)

wavres1 = bl.continuous_wavelet_transform(times1, counts1, 
                                         shortestperiod=10, longestperiod = 40)
wavres2 = bl.continuous_wavelet_transform(times2, counts2, 
                                         shortestperiod=10, longestperiod = 40)
wavres3 = bl.continuous_wavelet_transform(times3, counts3, 
                                         shortestperiod=10, longestperiod = 40)


names = ['Channel 1', 'Channel 2', 'Channel 9']

wavres=wavres1

plt.figure(figsize=(3.25, 6.))
gs = gridspec.GridSpec(2,1)
ax = plt.subplot(gs[0,0])
colors = ax.pcolormesh(wavres['x']/24, wavres['tau'],wavres['cwt_abs'], 
          vmax=wavres['cwt_abs'].max(), vmin=wavres['cwt_abs'].min())
ax.plot(wavres['x']/24, wavres['period'], 'white', label = 'Ridge')
#plt.colorbar(colors)
leg = plt.legend()
for text in leg.get_texts():
    plt.setp(text, color = 'w')
ax.set_xticklabels([])
ax.set_ylabel(r"$\tau$ (h)")
ax.set_title('Channel2')

# plotting the phase progression
bx = plt.subplot(gs[1,0])
bx.plot(wavres['x']/24, np.unwrap(wavres['phase']), 'k', label = 'Phase')
bx.set_xlabel("Time (day)")
bx.set_ylabel("Unwrapped Phase (rad)")
plt.tight_layout()

pulse_timing1 = (15*60 + 24*60*np.array([4,5,6,7,8,9,10]))/60
pulse_timing2 = (10*60 + 24*60*np.arange(15,29))/60+5

wavs = [wavres1, wavres2, wavres3]



def angle_at_pulse(wavres, timing):
    """ returns phase angle at start of pulse """
    locs = []
    times = []
    for time in timing:
        locs.append(np.argmin((wavres['x']-time)**2))
    ret_pulse_phis = wavres['phase'][locs]
    ret_pulse_amps = np.array([np.max(wavres['cwt_relamp'][:,loc])
                        for loc in locs])
    times = np.array(timing)[np.where(ret_pulse_amps>0.05)]
    phis = ret_pulse_phis[np.where(ret_pulse_amps>0.05)]
    return times, phis, ret_pulse_amps

# all phase tracking
all_phases = []
each_24h = np.arange(1,29)*24
times1, phases1, amps1 = angle_at_pulse(wavres1, each_24h)
times2, phases2, amps2 = angle_at_pulse(wavres2, each_24h)
times3, phases3, amps3 = angle_at_pulse(wavres3, each_24h)


plt.figure()
plo.PlotOptions(ticks='in')
ax = plt.subplot()

ax.plot([4,11], 2*np.pi*np.array([10,10])/24-2, color='h', lw=2, label='Lo')
ax.plot([15,28], 2*np.pi*np.array([15,15])/24-2, color='j', lw=2, label='Hi/Lo')

#ax.plot(times1/24, np.unwrap(phases1), 'fx', mew=1, label='Channel1 (VIPCre/+;Ai32/+)')
ax.plot(times2/24, np.unwrap(phases2), 'i+', mew=1,label='Channel2 (VIPCre/+;Ai32/+)')
ax.plot(times3/24, np.unwrap(phases3), 'k.', mew=1, label='Channel9 (Control)')




#plo.format_2pi_axis(ax, y=True, x=False)
ax.set_xlabel('Day')
ax.set_ylabel('Phase Angle (rad)')
plt.tight_layout(**plo.layout_pad)
plt.legend()





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












