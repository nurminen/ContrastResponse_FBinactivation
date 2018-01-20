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
penetrs = np.array([2,3,4,6])
#penetrs = np.array([2])
root_folder = '/opt3/' + animal + '/'
resu_folder = '/home/lauri/projects/ContrastResponse_FBinactivation/results/'
thresholded_data_file = tb.open_file(resu_folder + 'thresholded_CRF_data.h5','a')
# iterate penetrations
this_unit = 0
for p in penetrs:
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
                
                AN = (stats.ttest_1samp(spikeCounts_NoL[10,:],np.mean(blankR))[1] < 0.05) and (np.max(np.mean(spikeCounts_NoL,axis=1)) > 5)
                if AN:
                    # if the cell is visually responsive put data to table
                    if thresholded_data_file.__contains__('/'+'unit_'+str(this_unit)) == True:
                        group = thresholded_data_file.getNode('/'+'unit_'+str(this_unit))
                    else:
                        group = thresholded_data_file.create_group('/'+'unit_'+str(this_unit),'unit node',createparents=True)

