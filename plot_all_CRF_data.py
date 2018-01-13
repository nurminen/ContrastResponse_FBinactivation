# iterate through the CRF data
# no data selection
# plots

# 2018-1-13

import matplotlib.pyplot as plt
import numpy as np
import tables as tb

animal  = 'MM378'
#penetrs = np.array([2,3,4,6])
penetrs = np.array([2])
root_folder = '/opt3/' + animal + '/'


# iterate penetrations
for p in penetrs:
    # open the HDF5-file that is collated over all penetrations
    # the units have been matched across sortings using waveform euclidean distance
    data_file = tb.open_file(root_folder + 'collated_data_P' + str(p) + '.h5')
    # iterate through units
    for u in data_file.iter_nodes('/MM378/P'+str(p)):
        # iterate through datasets as not all datasets are necessarily measured for
        # each unit
        for ds in u.keys():
            # if the experiment was contrast then do plotting
            if u[ds] == 'contrast':
                # run cool analysis
                
        

