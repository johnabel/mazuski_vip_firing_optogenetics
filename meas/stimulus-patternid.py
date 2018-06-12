# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:38:30 2017

@author: abel

Fourth attempt at hierarchical clustering of neuronal firing pattern.
Here, we take each spike and add any spikes within 1s after the last, and
norm it. Here we are also using DIFR.
"""

from __future__ import division

import os
from collections import OrderedDict
import numpy  as np
import scipy as sp
from sklearn import decomposition, cluster
import scipy.cluster.hierarchy as sch
from fastdtw import fastdtw
import brewer2mpl as colorbrewer

import PlotOptions as plo

from matplotlib import gridspec
import matplotlib.pyplot as plt

# data location
data_path = 'stimulus_data/'


# load all neurons into dicts
locations_key = OrderedDict()
hif_neurons = OrderedDict()
hif_counter = 0
lif_neurons = OrderedDict()
lif_counter = 0

# loop through to pull the data
# start with hif
hif_loc = data_path+'/hif/'
lif_loc = data_path+'/lif/'

# set up the hif ones
for filei in os.listdir(hif_loc):
    if filei[-4:] == '.npy':
        # process it
        data = np.sort(np.load(hif_loc+filei))
        dx = np.diff(data)
        hd = np.histogram(dx, bins = np.arange(0.,1.001,0.001))
        neuron_id = 'h'+str(hif_counter)
        locations_key[neuron_id] = hif_loc+filei
        hif_neurons[neuron_id] = hd[0]/np.max(hd[0])
        hif_counter+=1
        print len(data)/(data.max()-data.min())
    else:
        # it is not a waveform file
        pass

# set up the lif ones
for filei in os.listdir(lif_loc):
    if filei[-4:] == '.npy':
        # process it
        data = np.sort(np.load(lif_loc+filei))
        dx = np.diff(data)
        hd = np.histogram(dx, bins = np.arange(0.,1.001,0.001))
        neuron_id = 'l'+str(lif_counter)
        locations_key[neuron_id] = lif_loc+filei
        lif_neurons[neuron_id] = hd[0]/np.max(hd[0])
        lif_counter+=1
        print len(data)/(data.max()-data.min())
    else:
        # it is not a waveform file
        pass




hif_data = np.asarray(hif_neurons.values())
lif_data = np.asarray(lif_neurons.values())
bins = np.arange(0.,1.001,0.001)[1:]-0.0005
hif_difr = np.asarray([1/bins[np.argmax(hif_i)] for hif_i in hif_data])
lif_difr = np.asarray([1/bins[np.argmax(lif_i)] for lif_i in lif_data])


palatte = colorbrewer.get_map('Dark2','Qualitative',7)
colors = palatte.mpl_colors
plo.PlotOptions(ticks='in')
plt.figure(figsize = (3.5,1.7))
gs = gridspec.GridSpec(2,2, width_ratios=(3,1))

ax = plt.subplot(gs[0,0])
ax.plot(bins, hif_data[0], color=colors[1])
ax.set_xlim([0,1])
ax.set_xticklabels([])
ax.set_yticks([0,0.5,1.])
ax.set_ylim([0,1])

bx = plt.subplot(gs[1,0])
bx.plot(bins, lif_data[0], color=colors[2])
bx.set_xlim([0,1])
bx.set_yticks([0,0.5,1.])
bx.set_ylim([0,1])
bx.set_xlabel("ISI (s)")


cx = plt.subplot(gs[:,1])
cx. plot([0-0.06,0+0.06,0,0], hif_difr, ls='', marker ='o', color=colors[1])
cx. plot([1-0.1,1-0.033,1+0.033,1+0.1], lif_difr,
         ls='', marker ='o', color=colors[2])
cx.set_xlim([-0.5,1.5])
cx.set_xticks([0,1])
cx.set_xticklabels(['HIF','LIF'])
cx.set_ylabel('DIFR (HZ)')


plt.tight_layout()

#
#
#            PLOT RASTER FOR PATTERNS
#

hname = 'h1'
adata = np.sort(np.load(locations_key[hname]))
adx = np.diff(adata)

lname = 'l2'
bdata = np.sort(np.load(locations_key[lname]))

hif_raster = adata[6000:]-adata[6000]
hif_raster = hif_raster[np.where(hif_raster<60)[0]]
lif_raster = bdata[6000:]-bdata[6000]
lif_raster = lif_raster[np.where(lif_raster<60)[0]]


def raster(ax, event_times_list, color='k', **kwargs):
    """
    Creates a raster plot
    Parameters
    ----------
    event_times_list : iterable
                       a list of event time iterables
    color : string
            color of vlines
    Returns
    -------
    ax : an axis containing the raster plot
    """
    # assuming 60s
    event_times_list= event_times_list-event_times_list[0]
    et1 = event_times_list[np.where(event_times_list<15)]
    et2 = event_times_list[np.min(np.where(event_times_list>15)):
                           np.max(np.where(event_times_list<30))]
    et3 = event_times_list[np.min(np.where(event_times_list>30)):
                           np.max(np.where(event_times_list<45))]
    et4 = event_times_list[np.where(event_times_list>45)]
    lists_of_times = [et1, et2, et3, et4]
    for ith, trial in enumerate(lists_of_times):
        for i,et in enumerate(trial):
            plt.vlines(et-ith*15, .55+ith, 1.45+ith, color = color)
    plt.ylim(.5, len(lists_of_times) + .5)
    plt.yticks([])
    plt.xlim([0,15])

def ISIH_60s(ax, data, color='k'):
    """
    creates an ISIH for the 60s of data
    """
    dx = np.diff(data)
    hd = np.histogram(dx, bins = np.arange(0.,2.01,0.02))
    ax.plot(np.arange(0.0,2.,0.02)+0.01, hd[0]/np.max(hd[0]), color=color)
    ax.set_xlim([-0.01,1])
    ax.set_xticks([0,1])
    ax.set_ylim([0,1])#np.round(np.max(hd[0]/np.sum(hd[0])),1)+0.1])
    ax.set_yticks([0,1])#np.round(np.max(hd[0]/np.sum(hd[0])),1)+0.1])


palatte = colorbrewer.get_map('Dark2','Qualitative',3)
colors = palatte.mpl_colors
plt.figure(figsize=(3.4,1.2))
gs = gridspec.GridSpec(2,2, width_ratios = (4,1))

ax = plt.subplot(gs[0,0])
raster(ax, hif_raster, color=colors[1])
ax.set_xticklabels([])
ax.set_ylabel('HIF')


bx = plt.subplot(gs[1,0])
raster(bx, lif_raster, color=colors[2])
bx.set_ylabel('LIF')
bx.set_xlabel('Time (s)')

cx = plt.subplot(gs[0,1])
cx.set_xticklabels([])
ISIH_60s(cx, hif_raster, colors[1])

dx = plt.subplot(gs[1,1])
ISIH_60s(dx, lif_raster, colors[2])


plt.tight_layout(**plo.layout_pad)

#
#
#            PLOT RASTER FOR PATTERNS
#


def raster_1s(ax, event_times_list, color='k', st=0, **kwargs):
    """
    Creates a raster plot
    Parameters
    ----------
    event_times_list : iterable
                       a list of event time iterables
    color : string
            color of vlines
    Returns
    -------
    ax : an axis containing the raster plot
    """
    # assuming 60s
    event_times_list= event_times_list
    et1 = event_times_list[np.min(np.where(event_times_list>st)):
                           np.max(np.where(event_times_list<st+5))]
    lists_of_times = [et1-et1[0]]
    for ith, trial in enumerate(lists_of_times):
        for i,et in enumerate(trial):
            plt.vlines(et, .55, 1.45, color = color)
    plt.ylim(.5, len(lists_of_times) + .5)
    plt.yticks([])
    plt.xlim([-0.05,1.0])
    plt.xticks([0.0, 0.5, 1.0])



colors = palatte.mpl_colors
plt.figure(figsize=(1.5,0.8))
gs = gridspec.GridSpec(2,1)

ax = plt.subplot(gs[0,0])
raster_1s(ax, hif_raster, st=45.5, color=colors[1])
ax.set_xticklabels([])
ax.set_ylabel('HIF')


bx = plt.subplot(gs[1,0])
raster_1s(bx, lif_raster, st=8, color=colors[2])
bx.set_ylabel('LIF')
bx.set_xlabel('Time (s)')


plt.tight_layout(**plo.layout_pad)
