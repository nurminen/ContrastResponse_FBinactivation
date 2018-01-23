# iterate through the CRF data
# no data selection
# plots

# 2018-1-13

import matplotlib.pyplot as plt
import numpy as np
import tables as tb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats

n_std = 4
animal  = 'MM378'
penetrs = np.array([4])
#penetrs = np.array([2])
root_folder = '/opt3/' + animal + '/'
resu_folder = '/home/lauri/projects/ContrastResponse_FBinactivation/results/'

laser_amps = []
# iterate penetrations
for p in penetrs:
    pdf = PdfPages(resu_folder + animal + '_P' + str(p) + '_ContrastResponse.pdf')
    # open the HDF5-file that is collated over all penetrations
    # the units have been matched across sortings using waveform euclidean distance
    data_file = tb.open_file(root_folder + 'collated_data_P' + str(p) + '.h5')
    # iterate through units
    for u in data_file.iter_nodes('/MM378/P'+str(p)):
        # iterate through datasets as not all datasets are necessarily measured for
        # each unit

        mk_page = False
        AN = False
        for ds in u._v_children.keys():
            # plot if the experiment was contrast then do plotting
            if u._v_children[ds].attrs['experiment'] == 'contrast':
                print(u._v_children[ds].attrs['laser_powr'])
                # gather data
                spikeCounts_NoL = np.squeeze(u._v_children[ds].col('spikeCounts_NoL'))
                spikeCounts_L   = np.squeeze(u._v_children[ds].col('spikeCounts_L'))
                spikeCounts_NoL_mean = np.mean(spikeCounts_NoL,axis=1)
                spikeCounts_L_mean   = np.mean(spikeCounts_L,axis=1)
                spikeCounts_NoL_SE = np.std(spikeCounts_NoL,axis=1)/np.sqrt(spikeCounts_NoL.shape[1])
                spikeCounts_L_SE   = np.std(spikeCounts_L,axis=1)/np.sqrt(spikeCounts_L.shape[1])
                blankR = u._v_children[ds].col('blankR')
                
                #AN = (stats.ttest_1samp(spikeCounts_NoL[10,:],np.mean(blankR))[1] < 0.05) and (np.max(np.mean(spikeCounts_NoL,axis=1)) > 5)
                AN = True
                if AN:
                    mk_page = True
                
                if u._v_children[ds].attrs['laser_powr'] == 800 and u._v_children[ds].attrs['stim_size'] == 1.2:
                    if AN:
                        plt.subplot(331)
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_NoL_mean,yerr=spikeCounts_NoL_SE,fmt='ko-')
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_L_mean,yerr=spikeCounts_L_SE,fmt='go-')
                        ax = plt.gca()
                        ax.axes.xaxis.set_ticklabels([])
                elif u._v_children[ds].attrs['laser_powr'] == 800 and u._v_children[ds].attrs['stim_size'] == 2.4:
                    if AN:
                        plt.subplot(332)
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_NoL_mean,yerr=spikeCounts_NoL_SE,fmt='ko-')
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_L_mean,yerr=spikeCounts_L_SE,fmt='go-')
                        ax = plt.gca()
                        ax.axes.xaxis.set_ticklabels([])
                elif u._v_children[ds].attrs['laser_powr'] == 800 and u._v_children[ds].attrs['stim_size'] == 3.6:
                    if AN:
                        plt.subplot(333)
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_NoL_mean,yerr=spikeCounts_NoL_SE,fmt='ko-')
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_L_mean,yerr=spikeCounts_L_SE,fmt='go-')
                        ax = plt.gca()
                        ax.axes.xaxis.set_ticklabels([])
                elif u._v_children[ds].attrs['laser_powr'] == 900 and u._v_children[ds].attrs['stim_size'] == 1.2:
                    if AN:
                        plt.subplot(334)
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_NoL_mean,yerr=spikeCounts_NoL_SE,fmt='ko-')
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_L_mean,yerr=spikeCounts_L_SE,fmt='go-')
                        ax = plt.gca()
                        ax.axes.xaxis.set_ticklabels([])
                elif u._v_children[ds].attrs['laser_powr'] == 900 and u._v_children[ds].attrs['stim_size'] == 2.4:
                    if AN:
                        plt.subplot(335)
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_NoL_mean,yerr=spikeCounts_NoL_SE,fmt='ko-')
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_L_mean,yerr=spikeCounts_L_SE,fmt='go-')
                        ax = plt.gca()
                        ax.axes.xaxis.set_ticklabels([])
                elif u._v_children[ds].attrs['laser_powr'] == 900 and u._v_children[ds].attrs['stim_size'] == 3.6:
                    if AN:
                        plt.subplot(336)
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_NoL_mean,yerr=spikeCounts_NoL_SE,fmt='ko-')
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_L_mean,yerr=spikeCounts_L_SE,fmt='go-')
                        ax = plt.gca()
                        ax.axes.xaxis.set_ticklabels([])
                elif u._v_children[ds].attrs['laser_powr'] == 1000 and u._v_children[ds].attrs['stim_size'] == 1.2:
                    if AN:
                        plt.subplot(337)
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_NoL_mean,yerr=spikeCounts_NoL_SE,fmt='ko-')
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_L_mean,yerr=spikeCounts_L_SE,fmt='go-')
                        ax = plt.gca()
                        ax.axes.xaxis.set_ticklabels([])
                elif u._v_children[ds].attrs['laser_powr'] == 1000 and u._v_children[ds].attrs['stim_size'] == 2.4:
                    if AN:
                        plt.subplot(338)
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_NoL_mean,yerr=spikeCounts_NoL_SE,fmt='ko-')
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_L_mean,yerr=spikeCounts_L_SE,fmt='go-')
                        ax = plt.gca()
                        ax.axes.xaxis.set_ticklabels([])
                elif u._v_children[ds].attrs['laser_powr'] == 1000 and u._v_children[ds].attrs['stim_size'] == 3.6:
                    if AN:
                        plt.subplot(339)
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_NoL_mean,yerr=spikeCounts_NoL_SE,fmt='ko-')
                        plt.errorbar(u._v_children[ds].attrs['vals'],spikeCounts_L_mean,yerr=spikeCounts_L_SE,fmt='go-')
                        ax = plt.gca()
                        ax.axes.xaxis.set_ticklabels([])

        if mk_page:                
            pdf.savefig()
        plt.clf()

    pdf.close()
