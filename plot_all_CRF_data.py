# iterate through the CRF data
# no data selection
# plots

# 2018-1-13

import matplotlib.pyplot as plt
import numpy as np
import tables as tb
import matplotlib.pyplot as plt

animal  = 'MM378'
#penetrs = np.array([2,3,4,6])
penetrs = np.array([2])
root_folder = '/opt3/' + animal + '/'

laser_amps = []
# iterate penetrations
for p in penetrs:
    # open the HDF5-file that is collated over all penetrations
    # the units have been matched across sortings using waveform euclidean distance
    data_file = tb.open_file(root_folder + 'collated_data_P' + str(p) + '.h5')
    # iterate through units
    for u in data_file.iter_nodes('/MM378/P'+str(p)):
        # iterate through datasets as not all datasets are necessarily measured for
        # each unit
        for ds in u._v_children.keys():
            # plot if the experiment was contrast then do plotting
            if u._v_children[ds].attrs['experiment'] == 'contrast':
                # gather data
                spikeCounts_NoL = np.squeeze(u._v_children[ds].col('spikeCounts_NoL'))
                spikeCounts_L   = np.squeeze(u._v_children[ds].col('spikeCounts_L'))
                spikeCounts_NoL_mean = np.mean(spikeCounts_NoL,axis=1)
                spikeCounts_L_mean   = np.mean(spikeCounts_L,axis=1)
                spikeCounts_NoL_SE = np.std(spikeCounts_NoL,axis=1)/np.sqrt(spikeCounts_NoL.shape[1])
                spikeCounts_L_SE   = np.std(spikeCounts_L,axis=1)/np.sqrt(spikeCounts_L.shape[1])
                
                if u._v_children[ds].attrs['laser_powr'] == 800 and u._v_children[ds].attrs['stim_size'] == 1.2:
                    plt.subplot(331)
                    plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_NoL_mean,yerr=spikeCounts_NoL_SE,fmt='go-')
                elif u._v_children[ds].attrs['laser_powr'] == 800 and u._v_children[ds].attrs['stim_size'] == 2.4:
                    plt.subplot(332)
                elif u._v_children[ds].attrs['laser_powr'] == 800 and u._v_children[ds].attrs['stim_size'] == 3.6:
                    plt.subplot(333)
                elif u._v_children[ds].attrs['laser_powr'] == 900 and u._v_children[ds].attrs['stim_size'] == 1.2:
                    plt.subplot(334)
                elif u._v_children[ds].attrs['laser_powr'] == 900 and u._v_children[ds].attrs['stim_size'] == 2.4:
                    plt.subplot(335)
                elif u._v_children[ds].attrs['laser_powr'] == 900 and u._v_children[ds].attrs['stim_size'] == 3.6:
                    plt.subplot(336)
                elif u._v_children[ds].attrs['laser_powr'] == 1000 and u._v_children[ds].attrs['stim_size'] == 1.2:
                    plt.subplot(337)
                elif u._v_children[ds].attrs['laser_powr'] == 1000 and u._v_children[ds].attrs['stim_size'] == 2.4:
                    plt.subplot(338)
                elif u._v_children[ds].attrs['laser_powr'] == 1000 and u._v_children[ds].attrs['stim_size'] == 3.6:
                    plt.subplot(339)
