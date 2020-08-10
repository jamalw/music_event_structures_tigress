import numpy as np
import sys
import nibabel as nib
from scipy.spatial import distance
import brainiak.eventseg.event
from scipy.stats import norm,zscore,pearsonr,stats
import os
from sklearn import linear_model
from srm import SRM_V1, SRM_V2, SRM_V3
import scipy.stats as st
import matplotlib.pyplot as plt

idx = int(sys.argv[1])
runNum = int(sys.argv[2])

def save_nifti(data,affine,savedir):
    minval = np.min(data)
    maxval = np.max(data)
    img = nib.Nifti1Image(data,affine)
    img.header["cal_min"] = minval
    img.header["cal_max"] = maxval
    img.header.set_data_dtype(np.float64)
    nib.save(img, savedir)
 
def HMM(X,Y,human_bounds):

    """fit hidden markov model
  
       Fit HMM to average data and cross-validate with leftout subject using within song and between song average correlations              

       Parameters
       ----------
       A: voxel by time ndarray (2D)
       B: voxel by time ndarray (2D)
       C: voxel by time ndarray (2D)
       D: voxel by time ndarray (2D)
       K: # of events for HMM (scalar)
 
       Returns
       -------
       z: z-score after performing permuted cross-validation analysis      

    """

    # Fit to all but one subject
    nPerm=1000
    within_across = np.zeros(nPerm + 1)
    K = len(human_bounds) + 1
    nTR = X.shape[1]
    ev = brainiak.eventseg.event.EventSegment(K)
    ev.fit(X.T)
    events = np.argmax(ev.segments_[0],axis=1)
    max_event_length = stats.mode(events)[1][0]

    # compute timepoint by timepoint correlation matrix
    cc = np.corrcoef(Y.T) # Should be a time by time correlation matrix

    # Create a mask to only look at values up to max_event_length
    local_mask = np.zeros(cc.shape, dtype=bool)
    for k in range(1,max_event_length):
        local_mask[np.diag(np.ones(cc.shape[0]-k, dtype=bool), k)] = True

    for p in range(nPerm+1):
        #match[p] = sum([np.min(np.abs(perm_bounds - hb)) for hb in human_bounds])
        #match[p] = np.sqrt(sum([np.min((perm_bounds - hb)**2) for hb in human_bounds]))
         
        same_event = events[:,np.newaxis] == events
        within = cc[same_event*local_mask].mean()
        across = cc[(~same_event)*local_mask].mean()
        within_across[p] = within - across

        np.random.seed(p)
        events = np.zeros(nTR, dtype=np.int)
        events[np.random.choice(nTR,K-1,replace=False)] = 1
        events = np.cumsum(events)

    return within_across


if runNum == 0:
    # run 1 times
    song_bounds = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973, 2198,2377,2511])

    songs = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

    # get song start time and end time for run 1
    start_idx = song_bounds[idx] 
    end_idx   = song_bounds[idx+1]

elif runNum == 1:
    # run 2 times
    song_bounds = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

    songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

    # get song start time and end time for run 2
    start_idx = song_bounds[idx]
    end_idx   = song_bounds[idx+1]

datadir = '/tigress/jamalw/MES/'

mask_img = nib.load(datadir + 'data/mask_nonan.nii')

n_iter = 50
srm_k = 30

# load human boundaries
human_bounds = np.load(datadir + 'prototype/link/scripts/data/beh/annotations/' + songs[idx] + '_beh_seg.npy')

parcels = nib.load(datadir + "data/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz").get_data()

# create brain-like object to store data into
#pvals = np.zeros_like(mask_img.get_data(),dtype=float)
#match = np.zeros_like(mask_img.get_data())

parcel_dir = datadir + "prototype/link/scripts/data/searchlight_input/parcels/Schaefer100/"

n_folds = 5
WvA = np.zeros((n_folds,1001))

for i in range(int(np.max(parcels))):
    print("Parcel Num: ", str(i+1))
    # get indices where mask and parcels overlap
    indices = np.where((mask_img.get_data() > 0) & (parcels == i + 1))
 
    # initialize list for storing masked data across subjects
    run1 = np.load(parcel_dir + "parcel" + str(i+1) + "_run1.npy")
    run2 = np.load(parcel_dir + "parcel" + str(i+1) + "_run2.npy")
    
    # run SRM on masked data
    if runNum == 0:
        shared_data = SRM_V1(run1,run2,srm_k,n_iter)
    elif runNum == 1:
        shared_data = SRM_V1(run2,run1,srm_k,n_iter)

    # perform cross-validation style HMM for n_folds
    for n in range(n_folds):
        np.random.seed(n)
        subj_list_shuffle = np.random.permutation(shared_data) 

        # convert data from list to numpy array and z-score in time
        shared_data_stack = stats.zscore(np.dstack(subj_list_shuffle),axis=1,ddof=1)

        # split subjects into two groups
        others = np.mean(shared_data_stack[:,start_idx:end_idx,:13],axis=2)    
        loo = np.mean(shared_data_stack[:,start_idx:end_idx,13:],axis=2)

        # fit HMM to song data and return match data where first entry is true match score and all others are permutation scores
        print("Fitting HMM")
        WvA[n,:] = HMM(others,loo,human_bounds)

    # take average of WvA scores over folds
    avgWvA = np.mean(WvA,axis=0)
         
    # compute z-score
    match_z = (avgWvA[0] - np.mean(avgWvA[1:])) / (np.std(avgWvA[1:]))
    
    # convert z-score to p-value
    match_p =  st.norm.sf(match_z)
 
    # compute p-value
    #match_p = (np.sum(SL_match[1:] <= SL_match[0]) + 1) / (len(SL_match))

    # fit wva score and pvalue into brain
    pvals[indices] = match_p  
    match[indices] = match_z 

savedir = "/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/parcels/Schaefer100/" + song_name

if  runNum == 0:
    pfn = savedir + "/pvals_srm_v1_test_run2"
    mfn = savedir + "/match_scores_srm_v1_test_run2"
elif runNum == 1:
    pfn = savedir + "/pvals_srm_v1_test_run1"
    mfn = savedir + "/match_scores_srm_v1_test_run1"

save_nifti(pvals, mask_img.affine, pfn) 
save_nifti(match, mask_img.affine, mfn)

