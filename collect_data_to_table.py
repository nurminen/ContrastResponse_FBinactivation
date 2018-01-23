# iterate through the CRF data
# no data selection
# plots

# 2018-1-22
import matplotlib.pyplot as plt
import sys
import numpy as np
import tables as tb
import matplotlib.pyplot as plt
import scipy.stats as stats
sys.path.append('/home/lauri/code/Libs')
import fitlib as fitlib
from scipy.optimize import minimize as sci_min 

n_std = 4
animal  = 'MM378'
penetrs = np.array([2,3,4,6])
#penetrs = np.array([2])
root_folder = '/opt3/' + animal + '/'
resu_folder = '/home/lauri/projects/ContrastResponse_FBinactivation/results/'
thresholded_data_file = tb.open_file(resu_folder + 'thresholded_CRF_data.h5','a')
group = thresholded_data_file.create_group('/', 'CRF', title='Contrast response data fitted with Naka-Rushton', createparents=True)

# give unique unit number across penetrations
this_unit = 0
create_table = True
for p in penetrs:
    # open the HDF5-file that is collated over all penetrations
    # the units have been matched across sortings using waveform euclidean distance
    data_file = tb.open_file(root_folder + 'collated_data_P' + str(p) + '.h5')
    # iterate through units
    for u in data_file.iter_nodes('/MM378/P'+str(p)):
        # iterate through datasets as not all datasets are necessarily measured for
        # each unit
        this_unit = this_unit + 1
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

                if create_table:
                    # make one table containin just the means and stds
                    dataTable = {'spikeCounts_NoL_mean':tb.Float64Col(shape=(spikeCounts_NoL_mean.shape[0],)),
                                 'spikeCounts_L_mean':tb.Float64Col(shape=(spikeCounts_L_mean.shape[0],)),
                                 'spikeCounts_NoL_SE':tb.Float64Col(shape=(spikeCounts_NoL_mean.shape[0],)),
                                 'spikeCounts_L_SE':tb.Float64Col(shape=(spikeCounts_L_mean.shape[0],)),
                                 'blankR':tb.Float64Col(shape=(blankR.shape[0],)),
                                 'case':tb.StringCol(5),
                                 'unit':tb.Float64Col(shape=(1,)),
                                 'layer':tb.StringCol(5),
                                 'C50':tb.Float64Col(shape=(1,)),
                                 'Rmax':tb.Float64Col(shape=(1,)),
                                 'n':tb.Float64Col(shape=(1,)),
                                 'C50_L':tb.Float64Col(shape=(1,)),
                                 'Rmax_L':tb.Float64Col(shape=(1,)),
                                 'n_L':tb.Float64Col(shape=(1,)),
                                 'diameter':tb.Float64Col(shape=(1,)),
                                 'laser_power':tb.Float64Col(shape=(1,)),
                                 'contrast_values':tb.Float64Col(shape=(u._v_children[ds].attrs['vals'].shape[0],))}


                    dats = thresholded_data_file.create_table(group,'Data',dataTable,'Fitted contrast response data')
                    create_table = False
                
                max_ind = np.where(spikeCounts_NoL_mean == np.max(spikeCounts_NoL_mean))[0]
                AN = (stats.ttest_1samp(spikeCounts_NoL[max_ind[0],:],np.mean(blankR))[1]) and (np.max(np.mean(spikeCounts_NoL,axis=1)) > (np.mean(blankR) + 5.0))
                if AN:
                    # fit control data
                    contrast = 100.0 * u._v_children[ds].attrs['vals']
                    response = spikeCounts_NoL_mean - np.mean(blankR)
                    args = (contrast, response)
                    bnds = ((0,None), (np.max(response),np.max(response)), (0,None))
                    # prior guess for parameters
                    par_prior = [30, 30, 2]
                    res = sci_min(fitlib.NakaRushton, par_prior, method='L-BFGS-B', args=args,bounds=bnds)
                    # fit laser data
                    response = spikeCounts_L_mean - np.mean(blankR)
                    bnds = ((0,None), (np.max(response),np.max(response)), (0,None))
                    args = (contrast, response)
                    res_L = sci_min(fitlib.NakaRushton, par_prior, method='L-BFGS-B', args=args,bounds=bnds)
                    
                    dats_row = dats.row
                    dats_row['spikeCounts_NoL_mean'] = spikeCounts_NoL_mean
                    dats_row['spikeCounts_L_mean']   = spikeCounts_L_mean
                    dats_row['spikeCounts_NoL_SE']   = spikeCounts_NoL_SE
                    dats_row['spikeCounts_L_SE']     = spikeCounts_L_SE
                    dats_row['blankR']   = np.mean(blankR)
                    dats_row['case']     = animal + 'P'+ str(p)
                    dats_row['diameter'] = u._v_children[ds].attrs['stim_size']
                    dats_row['laser_power'] = u._v_children[ds].attrs['laser_powr']
                    dats_row['unit'] = this_unit
                    dats_row['contrast_values'] = u._v_children[ds].attrs['vals']
                    dats_row['C50']  = res.x[0]
                    dats_row['Rmax'] = res.x[1]
                    dats_row['n']    = res.x[2]
                    dats_row['C50_L']  = res_L.x[0]
                    dats_row['Rmax_L'] = res_L.x[1]
                    dats_row['n_L']    = res_L.x[2]

                    dats_row.append()
                    dats.flush()
                    

thresholded_data_file.close()
data_file.close()
