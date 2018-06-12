# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:38:30 2017

@author: abel

Here we are plotting the ISI so I can check to make sure that the hierarchical
clustering does what we expect it to.
"""

from __future__ import division

import os
from collections import OrderedDict
import numpy  as np
import scipy as sp
from sklearn import decomposition, cluster
import scipy.cluster.hierarchy as sch
from fastdtw import fastdtw

import PlotOptions as plo

from matplotlib import gridspec
import matplotlib.pyplot as plt

# data location
data_path = 'sorted_data/'
# identify all the folders containing data
meas = []
for diri in os.listdir(data_path):
    if os.path.isdir(data_path+diri):
        meas.append(diri)

# load all neurons into dicts
locations_key = OrderedDict()
vip_neurons = OrderedDict()
vip_counter = 0
nonvip_neurons = OrderedDict()
nonvip_counter = 0
# loop through to pull the data
for diri in meas:
    # start with VIP neurons
    vip_loc = data_path+diri+'/VIP/'
    nonvip_loc = data_path+diri+'/NonVIP/'

    # set up the VIP ones
    for filei in os.listdir(vip_loc):
        if filei[-4:] == '.npy':
            # process it
            data = np.sort(np.load(vip_loc+filei))
            neuron_id = 'v'+str(vip_counter)
            locations_key[neuron_id] = vip_loc+filei
            vip_neurons[neuron_id] = data
            vip_counter+=1
        else:
            # it is not a waveform file
            pass

    # set up the nonVIP ones
    for filei in os.listdir(nonvip_loc):
        if filei[-4:] == '.npy':
            # process it
            data = np.sort(np.load(nonvip_loc+filei))
            neuron_id = 'n'+str(nonvip_counter)
            locations_key[neuron_id] = nonvip_loc+filei
            nonvip_neurons[neuron_id] = data
            nonvip_counter+=1
        else:
            # it is not a waveform file
            pass




vip_data = vip_neurons.values()
nonvip_data = nonvip_neurons.values()


for name in vip_neurons.keys():
    data = vip_neurons[name]
    dx = np.diff(data)
    fig = plt.figure()
    plt.plot(data[:-1], dx, 'k.', alpha=0.01)
    plt.title(name)
    plt.ylim([0,2])
    fig.savefig('individual_isi_plots/'+name+'.png')
    plt.clf()
    plt.close(fig)

for name in nonvip_neurons.keys():
    data = nonvip_neurons[name]
    dx = np.diff(data)
    fig = plt.figure()
    plt.plot(data[:-1], dx, 'k.', alpha=0.01)
    plt.title(name)
    plt.ylim([0,2])
    fig.savefig('individual_isi_plots/'+name+'.png')
    plt.clf()
    plt.close(fig)



