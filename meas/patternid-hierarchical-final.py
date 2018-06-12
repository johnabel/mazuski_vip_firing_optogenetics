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


# parameters for clustering
color_thresh = 0.7 #threshold for separate clusters: default
link_method = 'complete' #see scipy hierarchical clustering for details
binsize=0.015 # binsize of the isih
difr_weight = 0.0 # relative weighting of difr differences

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
            dx = np.diff(data)
            hd = np.histogram(dx, bins = np.arange(0.,1.51,binsize))
            neuron_id = 'v'+str(vip_counter)
            locations_key[neuron_id] = vip_loc+filei
            vip_neurons[neuron_id] = np.hstack([hd[0]/np.sum(hd[0]),
                                 1E-12])
            vip_counter+=1
        else:
            # it is not a waveform file
            pass

    # set up the nonVIP ones
    for filei in os.listdir(nonvip_loc):
        if filei[-4:] == '.npy':
            # process it
            data = np.sort(np.load(nonvip_loc+filei))
            dx = np.diff(data)
            hd = np.histogram(dx, bins = np.arange(0.,1.51,binsize))
            neuron_id = 'n'+str(nonvip_counter)
            locations_key[neuron_id] = nonvip_loc+filei
            nonvip_neurons[neuron_id] = np.hstack([hd[0]/np.sum(hd[0]),
                                 0])
            nonvip_counter+=1
        else:
            # it is not a waveform file
            pass




vip_data = np.asarray(vip_neurons.values())
nonvip_data = np.asarray(nonvip_neurons.values())
full_data = np.vstack([nonvip_data,vip_data])


def fdtw_dist(d1,d2):
    return fastdtw(d1,d2)[0]



#fastdtw vip data
D = np.copy(full_data)
y = sch.linkage(D, metric=fdtw_dist, method=link_method)
z = sch.dendrogram(y, orientation='right', no_plot=True,
    color_threshold=color_thresh*np.max(y[:,2]))
index = z['leaves']
D = D[index,:]
resulting_clusters = sch.fcluster(y, color_thresh*np.max(y[:,2]),
    criterion='distance')

palatte = colorbrewer.get_map('Set2','Qualitative',3)
colors = palatte.mpl_colors


plo.PlotOptions(ticks='in')
# panels a and b
plt.figure(figsize=(4.0,2.6))
gs = gridspec.GridSpec(3,5, width_ratios = (1,0.05,0.2,0.7,0.7))

ax = plt.subplot(gs[:,0])
cbar = ax.pcolormesh(D[:,:-1], cmap='plasma', vmax=0.05)
cbar.set_zorder(-20)
ax.set_yticks([0,len(index)])
#ax.set_yticklabels(index)
ax.set_ylabel('All Recorded Cells')
ax.set_xlabel('Time (s)')
ax.set_xticks([0,100])
ax.set_xticklabels([0.,1.5])
plt.gca().invert_yaxis()
ax.set_rasterization_zorder(-10)

aax = plt.subplot(gs[:,1])
cbar = aax.pcolormesh(D[:,-1:],cmap='gray_r',vmax=1E-12)
cbar.set_zorder(-20)
aax.set_yticks([])
aax.set_yticklabels([])
aax.set_xticklabels([])
aax.set_xticks([])
plt.gca().invert_yaxis()
aax.set_rasterization_zorder(-10)

bx = plt.subplot(gs[:,2])
z = sch.dendrogram(y, orientation='right', ax=bx,
                   color_threshold=color_thresh*np.max(y[:,2]))
bx.set_yticklabels([])
plt.gca().invert_yaxis()
bx.axis('off')

# separate broadly into categories
bur_count = list(resulting_clusters).count(1)
irr_count = list(resulting_clusters).count(3)
ton_count = list(resulting_clusters).count(2)
all_bur = D[:bur_count]
all_ton = D[bur_count:bur_count+ton_count]
all_irr = D[bur_count+ton_count:]

# the first 36 neurons are VIP+, the rest are VIP-
# sort into these groups
vip_bur = all_bur[np.where(all_bur[:,-1]==1E-12)[0]]
vip_irr = all_irr[np.where(all_irr[:,-1]==1E-12)[0]]
vip_ton = all_ton[np.where(all_ton[:,-1]==1E-12)[0]]

nip_bur = all_bur[np.where(all_bur[:,-1]==0)[0]]
nip_irr = all_irr[np.where(all_irr[:,-1]==0)[0]]
nip_ton = all_ton[np.where(all_ton[:,-1]==0)[0]]

# EXAMPLE VIP NEURONS

c1x = plt.subplot(gs[1,3])
c1x.plot(np.arange(0.0,1.5,0.015), vip_ton[3,:-1], color='f')
c1x.set_yticks([0,0.1])
c1x.set_ylim([0,.1])
c1x.set_xlim([-0.02,1.5])
c1x.set_xticks([0,.50,1,1.5])
c1x.set_xticklabels([])

c2x = plt.subplot(gs[2,3])
c2x.plot(np.arange(0.0,1.5,0.015), vip_irr[3,:-1], color='i')
c2x.set_yticks([0,0.05])
c2x.set_ylim([0,.05])
c2x.set_xlim([-0.02,1.5])
c2x.set_xticks([0,.50,1,1.5])
c2x.set_xticklabels([0,'','',1.5])

# EXAMPLE nonVIP NEURONS

d1x = plt.subplot(gs[0,4])
d1x.plot(np.arange(0.0,1.5,0.015), nip_bur[3,:-1], color='l')
d1x.set_yticks([0,.2])
d1x.set_ylim([0,.2])
d1x.set_xlim([-0.02,1.5])
d1x.set_xticks([0,.50,1,1.5])
d1x.set_xticklabels([])


d2x = plt.subplot(gs[1,4])
d2x.plot(np.arange(0.0,1.5,0.015), nip_ton[30,:-1], color='f')
d2x.set_yticks([0,0.1])
d2x.set_ylim([0,0.1])
d2x.set_xlim([-0.02,1.5])
d2x.set_xticks([0,.50,1,1.5])
d2x.set_xticklabels([])
d2x.set_yticklabels([])

d3x = plt.subplot(gs[2,4])
d3x.plot(np.arange(0.0,1.5,0.015), nip_irr[30,:-1], color='i')
d3x.set_yticks([0,0.05])
d3x.set_ylim([0,.05])
d3x.set_xlim([-0.02,1.5])
d3x.set_xticks([0,.50,1,1.5])
d3x.set_yticklabels([])
d3x.set_xticklabels([0,'','',1.5])


plt.savefig("figure_creating/test_rasterization.svg", dpi=900)

plt.figure(figsize = (2,1))
ax = plt.subplot()
plt.colorbar(cbar)








# part b - peak times.
# first, we discretize further so that we can get better precision on the peaks
# we can do this by ensuring each bin is equally spaced in frequency


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
    vip_dx = []
    non_dx = []
    # set up the VIP ones
    for filei in os.listdir(vip_loc):
        if filei[-4:] == '.npy':
            # process it
            data = np.sort(np.load(vip_loc+filei))
            dx = np.diff(data)
            hd = np.histogram(dx, bins = np.arange(0.,1.501,0.002))
            neuron_id = 'v'+str(vip_counter)
            locations_key[neuron_id] = vip_loc+filei
            vip_neurons[neuron_id] = hd[0]/np.sum(hd[0])#np.max(hd[0])
            vip_counter+=1
            vip_dx.append(dx)
        else:
            # it is not a waveform file
            pass

    # set up the nonVIP ones
    for filei in os.listdir(nonvip_loc):
        if filei[-4:] == '.npy':
            # process it
            data = np.sort(np.load(nonvip_loc+filei))
            dx = np.diff(data)
            hd = np.histogram(dx, bins = np.arange(0.,1.501,0.002))
            neuron_id = 'n'+str(nonvip_counter)
            locations_key[neuron_id] = nonvip_loc+filei
            nonvip_neurons[neuron_id] = hd[0]/np.sum(hd[0])#np.max(hd[0])
            nonvip_counter+=1
            non_dx.append(dx)
        else:
            # it is not a waveform file
            pass

# ALL isi's - combine to find dx for each pop
non_dx = np.hstack(non_dx)
vip_dx = np.hstack(vip_dx)

full_dx = np.hstack([non_dx, vip_dx])
hd = np.histogram(full_dx, bins = np.arange(0.,1.501,0.015))
tothd = np.sum(hd[0])
full_hist = hd[0]/np.sum(hd[0])

vhd = np.histogram(vip_dx, bins = np.arange(0.,1.501,0.015))
vip_hist = vhd[0]/np.sum(vhd[0])
nhd = np.histogram(non_dx, bins = np.arange(0.,1.501,0.015))
nonvip_hist = nhd[0]/np.sum(nhd[0])

plt.figure(figsize=(1.5,1.5))
ax= plt.subplot()
plt.plot(np.arange(0.,1.501,0.015)[:-1], full_hist,'k')
plt.plot(np.arange(0.,1.501,0.015)[:-1], nonvip_hist,'k',label='nonvip')
plt.plot(np.arange(0.,1.501,0.015)[:-1], vip_hist,'f',label='vip')
ax.axvline(0.05,color='f',ls=':')
ax.axvline(0.25,color='k',ls=':')
ax.axvline(0.5,color='f',ls=':')
plt.xlim([-0.01,1.5])
plt.ylim([0.0,0.02])
plt.yticks([0,0.005,0.01,0.015,0.02])
plt.legend()
plt.tight_layout(**plo.layout_pad)




plt.figure(figsize=(1.5,1.5))
ax= plt.subplot()
plt.plot(np.arange(0.,1.501,0.015)[:-1], nip_ton.mean(0)[:-1],'f:',label='nonVIP')
plt.plot(np.arange(0.,1.501,0.015)[:-1], nip_irr.mean(0)[:-1],'i:')
plt.plot(np.arange(0.,1.501,0.015)[:-1], nip_bur.mean(0)[:-1],'l:')

plt.plot(np.arange(0.,1.501,0.015)[:-1], vip_ton.mean(0)[:-1],'f',label='VIP')
plt.plot(np.arange(0.,1.501,0.015)[:-1], vip_irr.mean(0)[:-1],'i')
ax.axvline(0.05,color='f',ls='--')
ax.axvline(0.25,color='k',ls='--')
ax.axvline(0.45,color='f',ls='--')
plt.xlim([-0.01,1.5])
plt.ylim([0.0,0.05])
plt.yticks([0,0.05])
plt.legend()
plt.tight_layout(**plo.layout_pad)




vip_data = np.asarray(vip_neurons.values())
nonvip_data = np.asarray(nonvip_neurons.values())
full_data = np.vstack([nonvip_data,vip_data])

# get the data for vip, vip ton/irr, nonvip, nonvip ton/irr/bur
vip_count = len(vip_data)
nonvip_count = len(nonvip_data)
all_clust_ids = sch.fcluster(y, color_thresh*np.max(y[:,2]), 'distance')

nonvip_bur = np.where(all_clust_ids==1)[0][
                    np.where(all_clust_ids==1)[0]<nonvip_count]
nonvip_ton = np.where(all_clust_ids==2)[0][
                    np.where(all_clust_ids==2)[0]<nonvip_count]
nonvip_irr = np.where(all_clust_ids==3)[0][
                    np.where(all_clust_ids==3)[0]<nonvip_count]

vip_ton = np.where(all_clust_ids==2)[0][
                    np.where(all_clust_ids==2)[0]>=nonvip_count]
vip_irr = np.where(all_clust_ids==3)[0][
                    np.where(all_clust_ids==3)[0]>=nonvip_count]

# now, find the peak times
def pull_peak_times(data):
    """ finds the isi histogram peaks """
    bin_centers = np.arange(0.,1.501,0.002)
    data = np.asarray(data)
    maxs = np.argmax(data, axis=1)
    return bin_centers[maxs]

# vip data
tonic_vip = full_data[vip_ton, :]
irreg_vip = full_data[vip_irr, :]

# nonvip data
tonic_nonvip = full_data[nonvip_ton, :]
irreg_nonvip = full_data[nonvip_irr, :]
burst_nonvip = full_data[nonvip_bur, :]



#set up boxplot
vip_tonic_peaks = 1/pull_peak_times(tonic_vip)
vip_irreg_peaks = 1/pull_peak_times(irreg_vip)

nonvip_tonic_peaks = 1/pull_peak_times(tonic_nonvip)
nonvip_irreg_peaks = 1/pull_peak_times(irreg_nonvip)
nonvip_burst_peaks = 1/pull_peak_times(burst_nonvip)


boxplot_data = [vip_tonic_peaks,
                vip_irreg_peaks,
                nonvip_tonic_peaks,
                nonvip_irreg_peaks,
                nonvip_burst_peaks]

medians = [np.median(boxplot_data[i]) for i in range(len(boxplot_data))]
sems = [sp.stats.sem(boxplot_data[i]) for i in range(len(boxplot_data))]
stds = [np.std(boxplot_data[i]) for i in range(len(boxplot_data))]
iqrs = []
for datai in boxplot_data:
    q75, q25 = np.percentile(datai, [75 ,25])
    iqr = q75 - q25
    iqrs.append(iqr)

maxmin = [np.max(np.hstack(boxplot_data)), np.min(np.hstack(boxplot_data))]

names = ['VIP\nTON', 'VIP\nIRR', 'non-VIP\nTON',
         'non-VIP\nIRR', 'non-VIP\nBUR']

print medians, iqrs




plo.PlotOptions(ticks='in')
f, (ax2, ax) = plt.subplots(2, 1, sharex=True,
    figsize=(1.6,1.8))

box = ax.boxplot(boxplot_data, sym='', patch_artist=False, zorder=5)
colors = ['f','i','f','i','l']
alphs=[1,1,1,1,1]
for idx,datai in enumerate(boxplot_data):
    ax.scatter(np.array([idx+1]*len(datai))+(np.random.rand(len(datai))-0.5)*0.4,
                 datai, color=colors[idx], alpha=alphs[idx], s=0.5)
ax.set_ylim([0,20])

box = ax2.boxplot(boxplot_data, sym='', patch_artist=False)
for idx,datai in enumerate(boxplot_data):
    ax2.scatter([idx+1]*len(datai)+np.random.rand(len(datai))*0.2,
     datai, color=colors[idx], alpha=alphs[idx], s=0.5)
ax2.set_xticklabels(names)
ax2.set_ylim([30,400])

ax2.set_ylabel('Dominant Instantaneous\nFiring Freq. (DFIF, Hz)')

# hide the spines between ax and ax2
ax2.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.xaxis.tick_top()
ax2.tick_params(labeltop='off')  # don't put tick labels at the top
ax.xaxis.tick_bottom()

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


plt.tight_layout(**plo.layout_pad)
f.subplots_adjust(hspace=0.1)










#
# RASTER PLOTS
#
#



# show that patterns are unchanging throughout
irreg_unit = 'v'+str(vip_irr[2]-nonvip_count)
adata = np.sort(np.load(locations_key[irreg_unit]))
adx = np.diff(adata)

tonic_unit = 'v'+str(vip_ton[3]-nonvip_count)
bdata = np.sort(np.load(locations_key[tonic_unit]))
bdx = np.diff(bdata)

brsty_unit = 'n'+str(nonvip_bur[1])
cdata = np.sort(np.load(locations_key[brsty_unit]))
cdx = np.diff(cdata)

plt.figure(figsize=(3.5,1.0))
gs = gridspec.GridSpec(1,3)

ax = plt.subplot(gs[0,0])
ax.plot(adata[:-1][::2]/(3600*24), adx[::2], 'k.', alpha=0.002, zorder=-10)
ax.set_ylim([0,1])
ax.set_yticks([1,0])
ax.set_xlim([0,3])
ax.set_xticks([0,1,2,3])
ax.set_xticklabels(['0','','','3'])
ax.set_rasterization_zorder(-5)

bx = plt.subplot(gs[0,1])
bx.plot(bdata[:-1][::6]/(3600*24), bdx[::6], 'k.', alpha=0.002, zorder=-10)
bx.set_ylim([0,1])
bx.set_yticks([1,0])
bx.set_xlim([0,3])
bx.set_xticks([0,1,2,3])
bx.set_xticklabels(['0','','','3'])
bx.set_yticklabels(['','','',''])
bx.set_rasterization_zorder(-5)

cx = plt.subplot(gs[0,2])
cx.plot(cdata[:-1][::1]/(3600*24), cdx[::1], 'k.', alpha=0.002, zorder=-10)
cx.set_ylim([0,1])
cx.set_yticks([1,0])
cx.set_xlim([0,3])
cx.set_xticks([0,1,2,3])
cx.set_xticklabels(['0','','','3'])
cx.set_yticklabels(['','','',''])
cx.set_rasterization_zorder(-5)

ax.set_ylabel('ISI (s)')

bx.set_xlabel('Day')
plt.tight_layout(**plo.layout_pad)


plt.savefig("figure_creating/isi_rasterization.svg", dpi=600)















# raster plots for the example neurons shown
irreg_raster = adata[np.min(np.where(adata>1.4*24*60*60)[0]):
                        np.max(np.where(adata<1.4*24*60*60+60)[0])]
tonic_raster = bdata[np.min(np.where(bdata>2.05*24*60*60)[0]):
                        np.max(np.where(bdata<2.05*24*60*60+60)[0])]#1.15
brsty_raster = cdata[np.min(np.where(cdata>0.8*24*60*60-5.5)[0]):
                        np.max(np.where(cdata<0.8*24*60*60-5.5+60)[0])]

palatte = colorbrewer.get_map('Dark2','Qualitative',3)
colors = palatte.mpl_colors

def raster(ax, event_times_list, colors):
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
            plt.vlines(et-ith*15, .55+ith, 1.45+ith, color = colors[i%len(colors)])
    plt.ylim(.5, len(lists_of_times) + .5)
    plt.yticks([])
    plt.xlim([0,15])


plt.figure(figsize=(3.5,1.4))
gs = gridspec.GridSpec(3,1)

ax = plt.subplot(gs[0,0])
raster(ax, irreg_raster, colors=['i'])
ax.set_xticks([])
ax.set_ylabel('IRR')


bx = plt.subplot(gs[1,0])
raster(bx, tonic_raster, colors=['f'])
bx.set_xticks([])
bx.set_ylabel('TON')


cx = plt.subplot(gs[2,0])
raster(cx, brsty_raster, colors=['l'])
cx.set_ylabel('BUR')
cx.set_xlabel('Time (s)')

plt.tight_layout(**plo.layout_pad)



# check if 30Hz nonvip is bursty or tonic
# 31 HZ where nonvip_clusts==2 [0][116]; so n150
# 40hz where nonvip_clusts==2[0][11]; so n7
test_name = 'n17'


adata = np.load(locations_key[test_name])
adx = np.diff(adata)

# find out where to do raster
plt.figure(figsize=(3.5,1.2))
gs = gridspec.GridSpec(1,3)

ax = plt.subplot(gs[0,0])
ax.plot(adata[:-1][::1]/(3600*24), adx[::1], 'k.', alpha=0.002, zorder=-10)
ax.set_ylim([0,2])
ax.set_xlim([0,3])
ax.set_xticks([0,1,2,3])
ax.set_xticklabels(['0','','','3'])
ax.set_rasterization_zorder(-5)

# raster
test_raster = adata[:]
plt.figure(figsize=(3.5,0.8))
gs = gridspec.GridSpec(1,1)
ax = plt.subplot(gs[0,0])
raster(ax, test_raster)
plt.tight_layout()

"""
Figure notes:
a-
Used hierarchical clustering to identify patterns. Renormalized all isi
histograms, then used dynamic time warping with complete linkage to generate
clusters, with a thresh of d=0.7*dmax. Two emerged in each: one tonic, one
irregular. The Tonic patterns are characterized by a very sharp peak. The
tonic clusters also include "bursty" neurons, with a very high peak frequency
(a very short peak ISI). These neurons often fire very very rapidly in addition
to more slowly, but are not truly tonic, they form a third group.

A couple neurons are visually misclassified in each group -- this is generally
OK for unsupervised learning, and the idea is not sensitive to a perfect sort.

ISI hists calculated with a 0.01s binsize.

b -
Peak instantaneous frequency distribution. IRR < TON because it CAN have a
lower freq in addition to a higher freq and there is much more room for lower.
There are bursty outliers (peak >20hz) that fall into both groupings, and are
equally distributed between VIP/nonVIP. They are just neurons that fire in
very high frequency bursts, and are about ~15% of all neurons total.

Peak freq calculated by 1/peak ISI with the isi hist at 0.001s binsize for
increased precision for bursty neurons.

Bursty neurons removed from box plots as outliers.

c -
show that patterns are consistent throughout for (l) tonic (c) irregular (r)
bursty neurons. these are just plots of instantaneous ISI for 3 days
for tonic and irreg, only every third isi is plotted just for visualization
(otherwise the plot is just a big blur)

these are all VIP- just as an example

d -
raster plots showing pattern of each neuron's firing from the example

"""
