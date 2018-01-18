# iterate through the CRF data
# no data selection
# plots

# 2018-1-13

import matplotlib.pyplot as plt
import numpy as np
import tables as tb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as scistats

n_std = 4
animal  = 'MM378'
#penetrs = np.array([2,3,4,6])
penetrs = np.array([2])
root_folder = '/opt3/' + animal + '/'
resu_folder = '/home/lauri/projects/ContrastResponse_FBinactivation/results/'

pdf = PdfPages(resu_folder + animal + '_P' + str(penetrs) + '_ContrastResponse.pdf')
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
                blankR = u._v_children[ds].col('blankR')
                
                AN = (stats.ttest_1samp(spikeCounts_NoL[10,:],np.mean(blankR))[1] < 0.05) and (np.mean(spikeCounts_NoL[10,:]) > 5)

                if u._v_children[ds].attrs['laser_powr'] == 800 and u._v_children[ds].attrs['stim_size'] == 1.2:
                    plt.subplot(331)
                    if AN:
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_NoL_mean,yerr=spikeCounts_NoL_SE,fmt='ko-')
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_L_mean,yerr=spikeCounts_L_SE,fmt='go-')
                elif u._v_children[ds].attrs['laser_powr'] == 800 and u._v_children[ds].attrs['stim_size'] == 2.4:
                    plt.subplot(332)
                    if AN:
                        
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_NoL_mean,yerr=spikeCounts_NoL_SE,fmt='ko-')
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_L_mean,yerr=spikeCounts_L_SE,fmt='go-')
                elif u._v_children[ds].attrs['laser_powr'] == 800 and u._v_children[ds].attrs['stim_size'] == 3.6:
                    plt.subplot(333)
                    if AN:
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_NoL_mean,yerr=spikeCounts_NoL_SE,fmt='ko-')
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_L_mean,yerr=spikeCounts_L_SE,fmt='go-')
                elif u._v_children[ds].attrs['laser_powr'] == 900 and u._v_children[ds].attrs['stim_size'] == 1.2:
                    plt.subplot(334)
                    if AN:
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_NoL_mean,yerr=spikeCounts_NoL_SE,fmt='ko-')
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_L_mean,yerr=spikeCounts_L_SE,fmt='go-')
                elif u._v_children[ds].attrs['laser_powr'] == 900 and u._v_children[ds].attrs['stim_size'] == 2.4:
                    plt.subplot(335)
                    if AN:
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_NoL_mean,yerr=spikeCounts_NoL_SE,fmt='ko-')
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_L_mean,yerr=spikeCounts_L_SE,fmt='go-')
                elif u._v_children[ds].attrs['laser_powr'] == 900 and u._v_children[ds].attrs['stim_size'] == 3.6:
                    plt.subplot(336)
                    if AN:
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_NoL_mean,yerr=spikeCounts_NoL_SE,fmt='ko-')
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_L_mean,yerr=spikeCounts_L_SE,fmt='go-')
                elif u._v_children[ds].attrs['laser_powr'] == 1000 and u._v_children[ds].attrs['stim_size'] == 1.2:
                    plt.subplot(337)
                    if AN:
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_NoL_mean,yerr=spikeCounts_NoL_SE,fmt='ko-')
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_L_mean,yerr=spikeCounts_L_SE,fmt='go-')
                elif u._v_children[ds].attrs['laser_powr'] == 1000 and u._v_children[ds].attrs['stim_size'] == 2.4:
                    plt.subplot(338)
                    if AN:
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_NoL_mean,yerr=spikeCounts_NoL_SE,fmt='ko-')
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_L_mean,yerr=spikeCounts_L_SE,fmt='go-')
                elif u._v_children[ds].attrs['laser_powr'] == 1000 and u._v_children[ds].attrs['stim_size'] == 3.6:
                    plt.subplot(339)
                    if AN:
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_NoL_mean,yerr=spikeCounts_NoL_SE,fmt='ko-')
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_L_mean,yerr=spikeCounts_L_SE,fmt='go-')

        pdf.savefig()
        plt.clf()

pdf.close()
