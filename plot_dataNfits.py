import sys
import numpy as np
import tables as tb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.append('/home/lauri/code/Libs')
import fitlib as fitlib
import scipy.stats as stats

font = {'family': 'Sans',
        'color':  'black',
        'weight': 'bold',
        'size': 8,
        }

laser_label = {'family': 'Sans',
        'color':  'green',
        'weight': 'bold',
        'size': 6,
        }

label = {'family': 'Sans',
        'color':  'black',
        'weight': 'bold',
        'size': 6,
        }

resu_folder = '/home/lauri/projects/ContrastResponse_FBinactivation/results/'
thresholded_data_file = tb.open_file(resu_folder + 'thresholded_CRF_data.h5','a')
data = thresholded_data_file.root.CRF.Data

pdf = PdfPages(resu_folder + 'ContrastResponse_dataNakaRushton.pdf')
pdf_params = PdfPages(resu_folder + 'NakaRushton_params.pdf')

f, axarr = plt.subplots(1,2)
high_R2_rows = []
for i,r in enumerate(data.iterrows()):
    # R2 no laser
    res_NoL = r['spikeCounts_NoL_mean'] - r['blankR']
    R2_NoL  = stats.linregress(fitlib.NakaRushton(np.array([r['C50'], r['Rmax'], r['n']]),100 * r['contrast_values']),res_NoL)[2]
    # R2 laser
    res_L = r['spikeCounts_L_mean'] - r['blankR']
    R2_L  = stats.linregress(fitlib.NakaRushton(np.array([r['C50_L'], r['Rmax_L'], r['n_L']]),100 * r['contrast_values']),res_L)[2]

    if (R2_NoL >= 0.6) and (R2_L >= 0.6):
        high_R2_rows.append(i)
        contrast = 100 * np.linspace(r['contrast_values'][0],r['contrast_values'][-1],1000)
        # pull data WITHOUT laser
        fit_NoL = fitlib.NakaRushton(np.array([r['C50'], r['Rmax'], r['n']]),contrast)
        sem_NoL = r['spikeCounts_NoL_SE']
    
        # pull data WITH laser
        fit_L = fitlib.NakaRushton(np.array([r['C50_L'], r['Rmax_L'], r['n_L']]), contrast)

        sem_L = r['spikeCounts_L_SE']

        # plot LASER and NO-LASER in separate panels
        axarr[0].errorbar(100 * r['contrast_values'], res_NoL, yerr=sem_NoL, fmt='ko',figure=f)
        axarr[0].plot(contrast, fit_NoL, 'k-',figure=f)
        axarr[0].set_title(str(R2_NoL))
        #
        axarr[1].errorbar(100 * r['contrast_values'], res_L, yerr=sem_L, fmt='go',figure=f)
        axarr[1].plot(contrast, fit_L, 'g-',figure=f)
        axarr[1].set_title(str(R2_L))
        pdf.savefig()
        axarr[0].cla()
        axarr[1].cla()
    
pdf.close()

# extract data with R2 with and without laser > 0.6 and create new table for it
if thresholded_data_file.root.CRF.__contains__('high_R2_data'):
    high_R2_data = thresholded_data_file.root.CRF.high_R2_data
else:
    high_R2_data = data[high_R2_rows]
    high_R2_data = thresholded_data_file.create_table(thresholded_data_file.root.CRF,'high_R2_data',high_R2_data,'Fitted contrast response data')


# C50 
C50_small_NoL = [r['C50'][0] for r in high_R2_data.iterrows() if r['diameter'] == 1.2]
C50_small_L   = [r['C50_L'][0] for r in high_R2_data.iterrows() if r['diameter'] == 1.2]

C50_small_NoL = np.array(C50_small_NoL)
C50_small_L   = np.array(C50_small_L)

C50_near_NoL = [r['C50'][0] for r in high_R2_data.iterrows() if r['diameter'] == 2.4]
C50_near_L   = [r['C50_L'][0] for r in high_R2_data.iterrows() if r['diameter'] == 2.4]

C50_near_NoL = np.array(C50_near_NoL)
C50_near_L   = np.array(C50_near_L)

# Rmax
Rmax_small_NoL = [r['Rmax'][0] for r in high_R2_data.iterrows() if r['diameter'] == 1.2]
Rmax_small_L   = [r['Rmax_L'][0] for r in high_R2_data.iterrows() if r['diameter'] == 1.2]

Rmax_small_NoL = np.array(Rmax_small_NoL)
Rmax_small_L   = np.array(Rmax_small_L)

Rmax_near_NoL = [r['Rmax'][0] for r in high_R2_data.iterrows() if r['diameter'] == 2.4]
Rmax_near_L   = [r['Rmax_L'][0] for r in high_R2_data.iterrows() if r['diameter'] == 2.4]

Rmax_near_NoL = np.array(Rmax_near_NoL)
Rmax_near_L   = np.array(Rmax_near_L)

# n
n_small_NoL = [r['n'][0] for r in high_R2_data.iterrows() if r['diameter'] == 1.2]
n_small_L   = [r['n_L'][0] for r in high_R2_data.iterrows() if r['diameter'] == 1.2]

n_small_NoL = np.array(n_small_NoL)
n_small_L   = np.array(n_small_L)

# one value fitted to > 100, was removed as outlier
n_small_inds = np.where(n_small_NoL < 35)[0]
n_small_NoL  = n_small_NoL[n_small_inds]
n_small_L    = n_small_L[n_small_inds]

n_near_NoL = [r['n'][0] for r in high_R2_data.iterrows() if r['diameter'] == 2.4]
n_near_L   = [r['n_L'][0] for r in high_R2_data.iterrows() if r['diameter'] == 2.4]

n_near_NoL = np.array(n_near_NoL)
n_near_L   = np.array(n_near_L)

f, axarr = plt.subplots(2,3)
# stimulus in the RF
axarr[0,0].plot(C50_small_NoL,C50_small_L,'ko')
ax_max = 5 + np.max([axarr[0,0].get_ylim()[1], axarr[0,0].get_xlim()[1]])
ax_min = np.min([axarr[0,0].get_ylim()[0], axarr[0,0].get_xlim()[0]]) -1
axarr[0,0].plot([ax_min, ax_max],[ax_min, ax_max],'k-')
axarr[0,0].set_xlim(ax_min,ax_max)
axarr[0,0].set_ylim(ax_min,ax_max)
axarr[0,0].xaxis.set_visible(False)
axarr[0,0].set_ylabel('C50 laser',fontdict=laser_label)
axarr[0,0].set_aspect('equal','box')
axarr[0,0].set_title('C50 stimulus in RF',fontdict=font)

# stimulus in the RF
axarr[1,0].plot(C50_near_NoL,C50_near_L,'ko')
ax_max = 5 + np.max([axarr[1,0].get_ylim()[1], axarr[1,0].get_xlim()[1]])
ax_min = np.min([axarr[1,0].get_ylim()[0], axarr[1,0].get_xlim()[0]]) -1
axarr[1,0].plot([ax_min, ax_max],[ax_min, ax_max],'k-')
axarr[1,0].set_xlim(ax_min,ax_max)
axarr[1,0].set_ylim(ax_min,ax_max)
axarr[1,0].set_ylabel('C50 laser',fontdict=laser_label)
axarr[1,0].set_xlabel('C50 control',fontdict=label)
axarr[1,0].set_aspect('equal','box')
axarr[1,0].set_title('C50 stimulus \n in near-surround',fontdict=font)

# stimulus in the RF
axarr[0,1].plot(Rmax_small_NoL,Rmax_small_L,'ko')
ax_max = 5 + np.max([axarr[0,1].get_ylim()[1], axarr[0,1].get_xlim()[1]])
ax_min = np.min([axarr[0,1].get_ylim()[0], axarr[0,1].get_xlim()[0]]) -1
axarr[0,1].plot([ax_min, ax_max],[ax_min, ax_max],'k-')
axarr[0,1].set_xlim(ax_min,ax_max)
axarr[0,1].set_ylim(ax_min,ax_max)
axarr[0,1].xaxis.set_visible(False)
axarr[0,1].set_ylabel('Rmax laser',fontdict=laser_label)
axarr[0,1].set_aspect('equal','box')
axarr[0,1].set_title('Rmax stimulus in RF',fontdict=font)

# stimulus in the RF
axarr[1,1].plot(Rmax_near_NoL,Rmax_near_L,'ko')
ax_max = 5 + np.max([axarr[1,1].get_ylim()[1], axarr[1,1].get_xlim()[1]])
ax_min = np.max([axarr[1,1].get_ylim()[0], axarr[1,1].get_xlim()[0]]) - 1
axarr[1,1].plot([ax_min, ax_max],[ax_min, ax_max],'k-')
axarr[1,1].set_xlim(ax_min,ax_max)
axarr[1,1].set_ylim(ax_min,ax_max)
axarr[1,1].set_ylabel('Rmax laser',fontdict=laser_label)
axarr[1,1].set_xlabel('Rmax control',fontdict=label)
axarr[1,1].set_aspect('equal','box')
axarr[1,1].set_title('Rmax stimulus \n in near-surround',fontdict=font)

# n, stimulus in the RF
axarr[0,2].plot(n_small_NoL,n_small_L,'ko')
ax_max = 5 + np.max([axarr[0,2].get_ylim()[1], axarr[0,2].get_xlim()[1]])
ax_min = np.min([axarr[0,2].get_ylim()[0], axarr[0,2].get_xlim()[0]]) -1
axarr[0,2].plot([ax_min, ax_max],[ax_min, ax_max],'k-')
axarr[0,2].set_xlim(ax_min,ax_max)
axarr[0,2].set_ylim(ax_min,ax_max)
axarr[0,2].xaxis.set_visible(False)
axarr[0,2].set_ylabel('n laser',fontdict=laser_label)
axarr[0,2].set_aspect('equal','box')
axarr[0,2].set_title('n stimulus in RF',fontdict=font)

# n, stimulus encroaches to the near-surround
axarr[1,2].plot(n_near_NoL,n_near_L,'ko')
ax_max = 5 + np.max([axarr[1,2].get_ylim()[1], axarr[1,2].get_xlim()[1]])
ax_min = np.min([axarr[1,2].get_ylim()[0], axarr[1,2].get_xlim()[0]]) -1
axarr[1,2].plot([ax_min, ax_max],[ax_min, ax_max],'k-')
axarr[1,2].set_xlim(ax_min,ax_max)
axarr[1,2].set_ylim(ax_min,ax_max)
axarr[1,2].xaxis.set_visible(False)
axarr[1,2].set_ylabel('n laser',fontdict=laser_label)
axarr[1,2].set_aspect('equal','box')
axarr[1,2].set_title('n stimulus \n in near-surround',fontdict=font)

plt.tight_layout()
pdf_params.savefig()

f, axarr = plt.subplots(2,3)
# C50 stimulus confined to the RF
axarr[0,0].bar([1.0, 2.0], [np.mean(C50_small_NoL),np.mean(C50_small_L)],\
               1.0,yerr=[np.std(C50_small_NoL)/np.sqrt(C50_small_NoL.shape[0]),np.std(C50_small_L)/np.sqrt(C50_small_L.shape[0])],\
               color = ('grey','green'))
axarr[0,0].set_xlim(0, 3.0)
axarr[0,0].xaxis.set_visible(False)
axarr[0,0].set_aspect('auto','box')
axarr[0,0].set_title('C50 stimulus in RF',fontdict=font)

# # C50 stimulus encroaches to the near-surround 
axarr[1,0].bar([1.0, 2.0], [np.mean(C50_near_NoL),np.mean(C50_near_L)],\
               1.0,yerr=[np.std(C50_near_NoL)/np.sqrt(C50_near_NoL.shape[0]),np.std(C50_near_L)/np.sqrt(C50_near_L.shape[0])],\
               color = ('grey','green'))
axarr[1,0].set_xlim(0, 3.0)
axarr[1,0].set_aspect('auto','box')
axarr[1,0].set_title('C50 stimulus \n in near-surround',fontdict=font)

# Rmax stimulus confined to the RF
axarr[0,1].bar([1.0, 2.0], [np.mean(Rmax_small_NoL),np.mean(Rmax_small_L)],\
               1.0,yerr=[np.std(Rmax_small_NoL)/np.sqrt(Rmax_small_NoL.shape[0]),np.std(Rmax_small_L)/np.sqrt(Rmax_small_L.shape[0])],\
               color = ('grey','green'))
axarr[0,1].set_xlim(0, 3.0)
axarr[0,1].xaxis.set_visible(False)
axarr[0,1].set_aspect('auto','box')
axarr[0,1].set_title('Rmax stimulus in RF',fontdict=font)

# Rmax stimulus encroaches to the near-surround 
axarr[1,1].bar([1.0, 2.0], [np.mean(Rmax_near_NoL),np.mean(Rmax_near_L)],\
               1.0,yerr=[np.std(Rmax_near_NoL)/np.sqrt(Rmax_near_NoL.shape[0]),np.std(Rmax_near_L)/np.sqrt(Rmax_near_L.shape[0])],\
               color = ('grey','green'))
axarr[1,1].set_xlim(0, 3.0)
axarr[1,1].set_aspect('auto','box')
axarr[1,1].set_title('Rmax stimulus \n in near-surround',fontdict=font)

# n stimulus confined to the RF
axarr[0,2].bar([1.0, 2.0], [np.mean(n_small_NoL),np.mean(n_small_L)],\
               1.0,yerr=[np.std(n_small_NoL)/np.sqrt(n_small_NoL.shape[0]),np.std(n_small_L)/np.sqrt(n_small_L.shape[0])],\
               color = ('grey','green'))
axarr[0,2].set_xlim(0, 3.0)
axarr[0,2].xaxis.set_visible(False)
axarr[0,2].set_aspect('auto','box')
axarr[0,2].set_title('n stimulus in RF',fontdict=font)

# n stimulus encroaches to the near-surround 
axarr[1,2].bar([1.0, 2.0], [np.mean(n_near_NoL),np.mean(n_near_L)],\
               1.0,yerr=[np.std(n_near_NoL)/np.sqrt(n_near_NoL.shape[0]),np.std(n_near_L)/np.sqrt(n_near_L.shape[0])],\
               color = ('grey','green'))
axarr[1,2].set_xlim(0, 3.0)
axarr[1,2].set_aspect('auto','box')
axarr[1,2].set_title('n stimulus \n in near-surround',fontdict=font)

pdf_params.savefig()

# close files
pdf_params.close()
thresholded_data_file.close()
