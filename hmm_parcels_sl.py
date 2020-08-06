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
 
def HMM(X,human_bounds):

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
    w = 3
    K = len(human_bounds) + 1
    ev = brainiak.eventseg.event.EventSegment(K)
    ev.fit(X.T)
    bounds = np.where(np.diff(np.argmax(ev.segments_[0],axis=1)))[0]
    match = np.zeros(nPerm+1)
    events = np.argmax(ev.segments_[0],axis=1)
    _, event_lengths = np.unique(events, return_counts=True)
    perm_bounds = bounds.copy()
    nTR = X.shape[1] 

    for p in range(nPerm+1):
        #match[p] = sum([np.min(np.abs(perm_bounds - hb)) for hb in human_bounds])
        #match[p] = np.sqrt(sum([np.min((perm_bounds - hb)**2) for hb in human_bounds]))
        for hb in human_bounds:
            if np.any(np.abs(perm_bounds - hb) <= w):
                match[p] += 1
        match[p] /= len(human_bounds)
 
        np.random.seed(p)
        perm_lengths = np.random.permutation(event_lengths)
        events = np.zeros(nTR, dtype=np.int)
        events[np.cumsum(perm_lengths[:-1])] = 1
        # pick number of timepoints to rotate by  
        nrot = np.random.randint(len(events))
        # convert events to list to allow combining lists in next step
        events_lst = list(events)
        # rotate boundaries
        events_rot = np.array(events_lst[-nrot:] + events_lst[:-nrot])
        # select indexes for new boundaries
        perm_bounds = np.where(events_rot == 1)[0]

    return match

hrf = 4

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
human_bounds = np.load(datadir + 'prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/' + songs[idx] + '/' + songs[idx] + '_beh_seg.npy') + hrf

parcels = nib.load(datadir + "data/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.nii.gz").get_data()

# create brain-like object to store data into
#pvals = np.zeros_like(mask_img.get_data(),dtype=float)
#match = np.zeros_like(mask_img.get_data())

parcel_dir = datadir + "prototype/link/scripts/data/searchlight_input/parcels/Schaefer200/"

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

    # get song data from each run
    data = shared_data[:,start_idx:end_idx]

    # average song data between two runs
    data = (data_run1 + data_run2) / 2

    # fit HMM to song data and return match data where first entry is true match score and all others are permutation scores
    print("Fitting HMM")
    SL_match = HMM(data,human_bounds)
   
    # compute z-score
    match_z = (SL_match[0] - np.mean(SL_match[1:])) / (np.std(SL_match[1:]))
    
    # convert z-score to p-value
    match_p =  st.norm.sf(match_z)
 
    # compute p-value
    #match_p = (np.sum(SL_match[1:] <= SL_match[0]) + 1) / (len(SL_match))

    # fit match score and pvalue into brain
    pvals[indices] = match_p  
    match[indices] = match_z 

savedir = "/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/parcels/Schaefer200/" + song_name

pfn = savedir + "/pvals_srm_v3"
mfn = savedir + "/match_scores_srm_v3"

save_nifti(pvals, mask_img.affine, pfn) 
save_nifti(match, mask_img.affine, mfn)

