import numpy as np
import brainiak.eventseg.event
from scipy.stats import norm,zscore,pearsonr,stats
from nilearn.image import load_img
import sys
from brainiak.funcalign.srm import SRM
import nibabel as nib
import os
from scipy.spatial import distance

subjs = ['MES_022817_0','MES_030217_0','MES_032117_1','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0','MES_041417_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

song_bounds = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

song_idx = int(sys.argv[1])
n_folds = 20
hrf = 5
srm_k = 5
datadir = '/tigress/jamalw/MES/'
mask_img = load_img(datadir + 'data/mask_nonan.nii.gz')
mask = mask_img.get_data()
mask_reshape = np.reshape(mask,(91*109*91))

results_z = np.zeros((91,109,91))
results_real = np.zeros((91,109,91))
#results_perms = np.zeros((91,109,91,1001,n_folds)) 

human_bounds = np.load(datadir + 'prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_wva/' + songs[song_idx] + '/' + songs[song_idx] + '_beh_seg.npy') + hrf

def searchlight(coords,human_bounds,mask,song_idx,song_bounds,subjs,hrf,srm_k):
    
    """run searchlight 
       Create searchlight object and perform voxel function at each searchlight location
    
       Parameters
       ----------
       data1  : voxel by time ndarray (2D); leftout subject run 1
       data2  : voxel by time ndarray (2D); average of others run 1
       data3  : voxel by time ndarray (2D); leftout subject run 2
       data4  : voxel by time ndarray (2D); average of others run 2
       coords : voxel by xyz ndarray (2D, Vx3)
       K      : # of events for HMM (scalar)
       
       Returns
       -------
       3D data: brain (or ROI) filled with searchlight function scores (3D)
    """

    stride = 5
    radius = 5
    min_vox = srm_k
    nPerm = 1000
    SL_allvox = []
    SL_results = []
    datadir = '/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_input/'
    for x in range(0,np.max(coords, axis=0)[0]+stride,stride):
        for y in range(0,np.max(coords, axis=0)[1]+stride,stride):
           for z in range(0,np.max(coords, axis=0)[2]+stride,stride):
               if not os.path.isfile(datadir + subjs[0] + '/' + str(x) + '_' + str(y) + '_' + str(z) + '.npy'):
                   continue
               D = distance.cdist(coords,np.array([x,y,z]).reshape((1,3)))[:,0]
               SL_vox = D <= radius
               data = []
               for i in range(len(subjs)):
                   subj_data = np.load(datadir + subjs[i] + '/' + str(x) + '_' + str(y) + '_' + str(z) + '.npy')
                   data.append(np.nan_to_num(stats.zscore(subj_data[:,:,0],axis=1,ddof=1)))
               for i in range(len(subjs)):
                   subj_data = np.load(datadir + subjs[i] + '/' + str(x) + '_' + str(y) + '_' + str(z) + '.npy')
                   data.append(np.nan_to_num(stats.zscore(subj_data[:,:,1],axis=1,ddof=1))) 
               print("Running Searchlight")
               # only run function on searchlights with #of voxels greater than or equal to min_vox
               if data[0].shape[0] >= min_vox:
                   SL_within_across = HMM(data,human_bounds,song_idx,song_bounds,hrf,srm_k)
                   SL_results.append(SL_within_across)
                   SL_allvox.append(np.array(np.nonzero(SL_vox)[0])) 
    voxmean = np.zeros((coords.shape[0], nPerm+1))
    vox_SLcount = np.zeros(coords.shape[0])
    for sl in range(len(SL_results)):
       voxmean[SL_allvox[sl],:] += SL_results[sl]
       vox_SLcount[SL_allvox[sl]] += 1
    voxmean = voxmean / vox_SLcount[:,np.newaxis]
    vox_z = np.zeros((coords.shape[0], nPerm+1))
    for p in range(nPerm+1):
        vox_z[:,p] = (voxmean[:,p] - np.mean(voxmean[:,1:],axis=1))/np.std(voxmean[:,1:],axis=1) 
    return vox_z,voxmean

def HMM(X,human_bounds,song_idx,song_bounds,hrf,srm_k):
    
    """fit hidden markov model
  
       Fit HMM to average data and cross-validate with leftout subjects using within song and between song average correlations              
       Parameters
       ----------
       A: list of 50 (contains 2 runs per subject) 2D (voxels x full time course) arrays
       B: # of events for HMM (scalar)
       song_idx: song index (scalar)
       C: voxel by time ndarray (2D)
       D: array of song boundaries (1D)
 
       Returns
       -------
       wVa score: final score after performing cross-validation of leftout subjects      
    """
    
    w = 6
    nPerm = 1000
    within_across = np.zeros(nPerm+1)
    run1 = [X[i] for i in np.arange(0, int(len(X)/2))]
    run2 = [X[i] for i in np.arange(int(len(X)/2), len(X))]
    print('Building Model')
    srm = SRM(n_iter=10, features=srm_k)   
    print('Training Model')
    srm.fit(run1)
    print('Testing Model')
    shared_data = srm.transform(run2)
    shared_data = stats.zscore(np.dstack(shared_data),axis=1,ddof=1)
    others = np.mean(shared_data[:,song_bounds[song_idx]:song_bounds[song_idx + 1],:13],axis=2)
    loo = np.mean(shared_data[:,song_bounds[song_idx]:song_bounds[song_idx + 1],13:],axis=2) 
    nTR = loo.shape[1]

    # Fit to all but one subject
    K = len(human_bounds) + 1
    ev = brainiak.eventseg.event.EventSegment(K)
    ev.fit(others.T)
    events = np.argmax(ev.segments_[0],axis=1)

    # Compute correlations separated by w in time
    corrs = np.zeros(nTR-w)
    for t in range(nTR-w):
        corrs[t] = pearsonr(loo[:,t],loo[:,t+w])[0]

    # Compute within vs across boundary correlations, for real and permuted bounds
    for p in range(nPerm+1):
        within = corrs[events[:-w] == events[w:]].mean()
        across = corrs[events[:-w] != events[w:]].mean()
        within_across[p] = within - across

        np.random.seed(p)
        events = np.zeros(nTR, dtype=np.int)
        events[np.random.choice(nTR,K-1,replace=False)] = 1
        events = np.cumsum(events)
    print((within_across[0] - np.mean(within_across[1:]))/np.std(within_across[1:]))
    return within_across


for i in range(n_folds):
    # create coords matrix
    results3d = np.zeros((91,109,91))
    results3d_real = np.zeros((91,109,91))
    #results3d_perms = np.zeros((91,109,91,1001))
    x,y,z = np.mgrid[[slice(dm) for dm in tuple((91,109,91))]]
    x = np.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2]))
    y = np.reshape(y,(y.shape[0]*y.shape[1]*y.shape[2]))
    z = np.reshape(z,(z.shape[0]*z.shape[1]*z.shape[2]))
    coords = np.vstack((x,y,z)).T 
    coords_mask = coords[mask_reshape>0]
    # permute subject IDs
    np.random.seed(i)
    subjs = np.random.permutation(subjs)  
    # prepare to run searchlight
    print('Running Distribute...')
    vox_z,raw_wVa_scores = searchlight(coords_mask,human_bounds,mask,song_idx,song_bounds,subjs,hrf,srm_k) 
    # store and average raw scores, z-scores, and permutations
    results3d[mask>0] = vox_z[:,0]
    results_z[:,:,:] += results3d/n_folds
    results3d_real[mask>0] = raw_wVa_scores[:,0]
    results_real[:,:,:] += results3d_real/n_folds
    #for j in range(vox_z.shape[1]):
    #    results3d_perms[mask>0,j] = vox_z[:,j]
    #results_perms[:,:,:,:,i] += results3d_perms/n_folds

# save results 
print('Saving to Searchlight Folders')
np.save('/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_wva/' + songs[song_idx] +'/real/full_brain/globals_K_raw_train_run1_reps_' + str(n_folds) + '_srm_k_' + str(srm_k) , results_real)
np.save('/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_wva/' + songs[song_idx] +'/zscores/full_brain/globals_K_zscores_train_run1_reps_' + str(n_folds) + '_srm_k_' + str(srm_k), results_z)
#np.save('/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/' + songs[song_idx] +'/perms/full_brain/globals_K_perms_' + str(K) + '_reps_' + str(n_folds), results_perms)
