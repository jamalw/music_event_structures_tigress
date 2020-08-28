from scipy.stats import norm,zscore,pearsonr,stats
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import sys
import nibabel as nib

idx = int(sys.argv[1])
runNum = int(sys.argv[2])

# Custom mean estimator with Fisher z transformation for correlations
def fisher_mean(correlation, axis=None):
    return np.tanh(np.mean(np.arctanh(correlation), axis=axis))

# set up directories
datadir = '/tigress/jamalw/MES/'
hmm_dir = datadir + "prototype/link/scripts/data/searchlight_output/parcels/Schaefer100/"
human_bounds_dir = datadir + "prototype/link/scripts/data/beh/annotations/"
parcel_dir = datadir + "prototype/link/scripts/data/searchlight_input/parcels/Schaefer100/"
savedir = datadir + "prototype/link/scripts/data/cross_corr_results/"

# load mask
mask_img = nib.load(datadir + 'data/mask_nonan.nii')

# set run 1 info
if runNum == 0:
    songs = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

    song_bounds = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973, 2198,2377,2511])

    start_idx = song_bounds[idx]
    end_idx = song_bounds[idx + 1]

# set run 2 info
elif runNum == 1:
    songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']
  
    song_bounds = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

    start_idx = song_bounds[idx]
    end_idx = song_bounds[idx + 1]

dur = end_idx - start_idx

n_iter = 50
srm_k = 30

parcels = nib.load(datadir + "data/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz").get_data()

# initialize empty timeseries
hmm_run1_series = np.zeros((dur))
hmm_run2_series = np.zeros((dur))
hb_series = np.zeros((dur)) 

# load hmm and human boundaries
hmm_bounds_run1 = np.around(np.load(hmm_dir + songs[idx] + '/hmm_bounds_rA1_run1.npy'))
hmm_bounds_run2 = np.around(np.load(hmm_dir + songs[idx] + '/hmm_bounds_rA1_run2.npy'))
human_bounds = np.load(human_bounds_dir + songs[idx] + '_beh_seg.npy')

# populate timeseries with boundaries
for b in range(len(human_bounds)):
    hmm_run1_series[int(hmm_bounds_run1[b])] = 1
    hmm_run2_series[int(hmm_bounds_run2[b])] = 1
    hb_series[int(human_bounds[b])] = 1   

for i in [60]:
    # get indice where mask and parcels overlap
    indices = np.where((mask_img.get_data() > 0) & (parcels == i + 1))
       
    if runNum == 0:
        data = np.load(parcel_dir + "parcel" + str(i+1) + "_run1.npy")
        data_avg = np.mean(np.mean(data[:,:,start_idx:end_idx],axis=0),axis=0)
    if runNum == 1:
        data = np.load(parcel_dir + "parcel" + str(i+1) + "_run2.npy")
        data_avg = np.mean(np.mean(data[:,:,start_idx:end_idx],axis=0),axis=0)
 
    cross_corr = signal.correlate(data_avg,hb_series,mode="full",method="direct")

    fig,ax = plt.subplots(figsize=(20,5))
    
    ax.plot(cross_corr,color='k')
    ax.axhline(0,0,dur*2)
    labels = np.arange(-dur,dur-1)
    x = np.arange(1,dur*2)
    middle_value = x[labels==0-1]
    label_lst = [labels[0], 0, labels[-1]+2]
    ax.set_xticks([x[0],middle_value,x[-1]+2])
    ax.set_xticklabels(label_lst)
    ax.set_title("cross-correlation for " + songs[idx],fontsize=15)
    plt.savefig(savedir + songs[idx] + "_rA1_human_" + "run" + str(runNum+1)) 
    
