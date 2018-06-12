# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:38:30 2017

@author: abel

This is for performing the same info as fig 2 for the collaborators data.
Both sets are VIP+.
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
data_path = 'collaborator_data/'
# identify all the folders containing data

# parameters for clustering
color_thresh = 0.7 #threshold for separate clusters
link_method = 'complete' #see scipy hierarchical clustering for details
binsize=0.05 # binsize of the isih
difr_weight = 0.0 # relative weighting of difr differences

# load all neurons into dicts
locations_key = OrderedDict()
wholecell_neurons = OrderedDict()
wholecell_counter = 0

# loop through to pull the data
wholecell_loc = data_path+'/Whole-cell_VIPrecordings/'

# set up the nonVIP ones
for filei in os.listdir(wholecell_loc):
    if filei[-4:] == '.csv':
        # process it
        data = np.sort(np.genfromtxt(wholecell_loc+filei))
        if len(data)>30:
            dx = np.diff(data)
            hd = np.histogram(dx, bins = np.arange(0.,1.51,binsize))
            neuron_id = 'w'+str(wholecell_counter)
            locations_key[neuron_id] = wholecell_loc+filei
            wholecell_neurons[neuron_id] =np.hstack([hd[0]/np.sum(hd[0]), 1E-12])
            wholecell_counter+=1
    else:
        # it is not a waveform file
        pass




wholecell_data = np.asarray(wholecell_neurons.values())

def fdtw_dist(d1,d2):
    return fastdtw(d1,d2)[0]



#fastdtw vip data
D = np.copy(wholecell_data)
y = sch.linkage(D, metric=fdtw_dist, method=link_method)
z = sch.dendrogram(y, orientation='right', no_plot=True,
                   color_threshold=color_thresh*np.max(y[:,2]))
index = z['leaves']
D = D[index,:]


palatte = colorbrewer.get_map('Set2','Qualitative',3)
colors = palatte.mpl_colors


plo.PlotOptions(ticks='in')
# panels a and b
plt.figure(figsize=(4.0,2.6))
gs = gridspec.GridSpec(3,5, width_ratios = (1,0.05,0.2,0.7,0.7))

ax = plt.subplot(gs[:,0])
cbar = ax.pcolormesh(D[:,:-1], cmap='plasma',vmax=0.5)
cbar.set_zorder(-20)
ax.set_yticks([0,len(index)])
#ax.set_yticklabels(index)
ax.set_ylabel('VIP Neurons')
ax.set_xlabel('ISI (s)')
ax.set_xticks([0,30])
ax.set_xticklabels([0.,1.5])
#plt.gca().invert_yaxis()
ax.set_rasterization_zorder(-10)

bx = plt.subplot(gs[:,2])
z = sch.dendrogram(y, orientation='right', ax=bx,
                   color_threshold=0.70*np.max(y[:,2]))
bx.set_yticklabels([])
#plt.gca().invert_yaxis()
bx.axis('off')


c1x = plt.subplot(gs[1,3])
c1x.plot(np.arange(0.0,1.5,0.05), D[50,:-1], color='f')
c1x.set_yticks([0,0.5])
c1x.set_ylim([0,0.5])
c1x.set_xlim([-0.02,1])
c1x.set_xticks([0,0.25,0.5,0.75,1.])
c1x.set_xticklabels([])

c2x = plt.subplot(gs[2,3])
c2x.plot(np.arange(0.0,1.5,0.05), D[12,:-1], color='i')
c2x.set_yticks([0,0.5])
c2x.set_ylim([0,0.5])
c2x.set_xlim([-0.02,1])
c2x.set_xticks([0,0.25,0.5,0.75,1.])
c2x.set_xticklabels([0,'','','',1])




plt.savefig("figure_creating/collab_rasterization.svg", dpi=900)



plt.tight_layout(**plo.layout_pad)

plt.figure(figsize = (2,1))
ax = plt.subplot()
plt.colorbar(cbar)








# part b - peak times.

# first, we discretize further so that we can get better precision on the peaks


# load all neurons into dicts
locations_key = OrderedDict()
wholecell_neurons = OrderedDict()
wholecell_counter = 0

wholecell_loc = data_path+'/Whole-cell_VIPrecordings/'


# set up the nonVIP ones
for filei in os.listdir(wholecell_loc):
    if filei[-4:] == '.csv':
        # process it
        data = np.sort(np.genfromtxt(wholecell_loc+filei))
        dx = np.diff(data)
        hd = np.histogram(dx, bins = np.arange(0.,1.01,0.02))
        neuron_id = 'w'+str(wholecell_counter)
        locations_key[neuron_id] = wholecell_loc+filei
        wholecell_neurons[neuron_id] = hd[0]/np.max(hd[0])
        wholecell_counter+=1
    else:
        # it is not a waveform file
        pass






wholecell_data = np.asarray(wholecell_neurons.values())


def pull_peak_times_wholecell(data):
    """ finds the isi histogram peaks """
    bin_centers = np.arange(0.,1.0,0.02)+0.01
    data = np.asarray(data)
    maxs = np.argmax(data, axis=1)
    return bin_centers[maxs]

# nonvip data, thresh ~62 = 0.7*dist_max
wholecell_clusts = sch.fcluster(y, 0.70*np.max(y[:,2]), 'distance')
tonic_ids = np.where(wholecell_clusts!=1)[0]
irreg_ids = np.where(wholecell_clusts==1)[0]
tonic_wholecell = wholecell_data[np.where(wholecell_clusts!=1)[0], :]
irreg_wholecell = wholecell_data[np.where(wholecell_clusts==1)[0], :]




#set up boxplot
wholecell_tonic_peaks = 1/pull_peak_times_wholecell(tonic_wholecell)
wholecell_irreg_peaks = 1/pull_peak_times_wholecell(irreg_wholecell)

boxplot_data = [
                wholecell_tonic_peaks,
                wholecell_irreg_peaks]
medians = [np.median(boxplot_data[i]) for i in range(len(boxplot_data))]
sems = [sp.stats.sem(boxplot_data[i]) for i in range(len(boxplot_data))]
stds = [np.std(boxplot_data[i]) for i in range(len(boxplot_data))]
iqrs = []
for datai in boxplot_data:
    q75, q25 = np.percentile(datai, [75 ,25])
    iqr = q75 - q25
    iqrs.append(iqr)
print medians,iqrs

names = ['Whole-Cell\nTON', 'Whole-Cell\nIRR']

plo.PlotOptions(ticks='in')
f = plt.figure(figsize=(1.8,1.8))
ax = plt.subplot()
box = ax.boxplot(boxplot_data, sym='.', patch_artist=True)
colors = ['f','i']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels(names)
ax.set_ylim([0,20])


plt.tight_layout(**plo.layout_pad)
f.subplots_adjust(hspace=0.1)













# raster plots for the example neurons shown
irreg_name = 'w'+str(irreg_ids[4])
adata = np.sort(np.genfromtxt(locations_key[irreg_name]))
adx = np.diff(adata)

tonic_name = 'w'+str(tonic_ids[1])
bdata = np.sort(np.genfromtxt(locations_key[tonic_name]))

irreg_raster = adata
tonic_raster = bdata


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


plt.figure(figsize=(3.5,0.9))
gs = gridspec.GridSpec(2,1)

ax = plt.subplot(gs[0,0])
raster(ax, irreg_raster)
ax.set_xticks([])
ax.set_ylabel('IRR')


bx = plt.subplot(gs[1,0])
raster(bx, tonic_raster)
bx.set_xticks([])
bx.set_ylabel('TON')

plt.tight_layout(**plo.layout_pad)

















"""
Figure notes:
Same exact methods as figure 2, with the only difference being for whole cell
recordings the bin sizes changed to deal with the lack of precision

"""

