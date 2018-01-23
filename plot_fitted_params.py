import numpy as np
import tables as tb
import matplotlib.pyplot as plt

resu_folder = '/home/lauri/projects/ContrastResponse_FBinactivation/results/'
thresholded_data_file = tb.open_file(resu_folder + 'thresholded_CRF_data.h5','a')
data = thresholded_data_file.root.CRF.Data



# C50 
C50_small_NoL = [r['C50'][0] for r in data.iterrows() if r['diameter'] == 1.2]
C50_small_L   = [r['C50_L'][0] for r in data.iterrows() if r['diameter'] == 1.2]

C50_small_NoL = np.array(C50_small_NoL)
C50_small_L   = np.array(C50_small_L)

C50_near_NoL = [r['C50'][0] for r in data.iterrows() if r['diameter'] == 2.4]
C50_near_L   = [r['C50_L'][0] for r in data.iterrows() if r['diameter'] == 2.4]

C50_near_NoL = np.array(C50_near_NoL)
C50_near_L   = np.array(C50_near_L)

# Rmax
Rmax_small_NoL = [r['Rmax'][0] for r in data.iterrows() if r['diameter'] == 1.2]
Rmax_small_L   = [r['Rmax_L'][0] for r in data.iterrows() if r['diameter'] == 1.2]

Rmax_small_NoL = np.array(Rmax_small_NoL)
Rmax_small_L   = np.array(Rmax_small_L)

Rmax_near_NoL = [r['Rmax'][0] for r in data.iterrows() if r['diameter'] == 2.4]
Rmax_near_L   = [r['Rmax_L'][0] for r in data.iterrows() if r['diameter'] == 2.4]

Rmax_near_NoL = np.array(Rmax_near_NoL)
Rmax_near_L   = np.array(Rmax_near_L)

# n
n_small_NoL = [r['n'][0] for r in data.iterrows() if r['diameter'] == 1.2]
n_small_L   = [r['n_L'][0] for r in data.iterrows() if r['diameter'] == 1.2]

n_small_NoL = np.array(n_small_NoL)
n_small_L   = np.array(n_small_L)

n_near_NoL = [r['n'][0] for r in data.iterrows() if r['diameter'] == 2.4]
n_near_L   = [r['n_L'][0] for r in data.iterrows() if r['diameter'] == 2.4]

n_near_NoL = np.array(n_near_NoL)
n_near_L   = np.array(n_near_L)

f, axarr = plt.subplots(2,2)
# stimulus in the RF
axarr[0,0].loglog(C50_small_NoL,C50_small_L,'ko')
axarr[0,0].loglog([0.1, 10e3],[0.1, 10e3],'k-')
ax_max = np.max([axarr[0,0].get_ylim()[1], axarr[0,0].get_xlim()[1]])
axarr[0,0].set_xlim(0.1,ax_max)
axarr[0,0].set_ylim(0.1,ax_max)
axarr[0,0].set_aspect('equal','box')

# stimulus in the RF
axarr[0,1].loglog(C50_near_NoL,C50_near_L,'ko')
axarr[0,1].loglog([0.1, 10e3],[0.1, 10e3],'k-')
ax_max = np.max([axarr[0,1].get_ylim()[1], axarr[0,1].get_xlim()[1]])
axarr[0,1].set_xlim(0.1,ax_max)
axarr[0,1].set_ylim(0.1,ax_max)
axarr[0,1].set_aspect('equal','box')

# stimulus in the RF
axarr[1,0].loglog(Rmax_small_NoL,Rmax_small_L,'ko')
axarr[1,0].loglog([0.1, 200],[0.1, 200],'k-')
ax_max = np.max([axarr[1,0].get_ylim()[1], axarr[1,0].get_xlim()[1]])
axarr[1,0].set_xlim(0.1,ax_max)
axarr[1,0].set_ylim(0.1,ax_max)
axarr[1,0].set_aspect('equal','box')

# stimulus in the RF
axarr[1,1].loglog(Rmax_near_NoL,Rmax_near_L,'ko')
axarr[1,1].loglog([0.1, 200],[0.1, 200],'k-')
ax_max = np.max([axarr[1,1].get_ylim()[1], axarr[1,1].get_xlim()[1]])
axarr[1,1].set_xlim(0.1,ax_max)
axarr[1,1].set_ylim(0.1,ax_max)
axarr[1,1].set_aspect('equal','box')

plt.show()
