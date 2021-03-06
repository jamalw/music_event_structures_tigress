import numpy as np
import brainiak.eventseg.event
from scipy.stats import norm,zscore,pearsonr,stats
from nilearn.image import load_img
import sys
from brainiak.funcalign.srm import SRM
import nibabel as nib
import os
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn import linear_model
from srm import SRM_V1, SRM_V2, SRM_V3

song_idx = int(sys.argv[1])
runNum = int(sys.argv[2])

subjs = ['MES_022817_0','MES_030217_0','MES_032117_1','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0','MES_041417_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

if runNum == 0:
    # run 1 times
    song_bounds = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973, 2198,2377,2511])

    songs = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

elif runNum == 1:
    # run 2 times
    song_bounds = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

    songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

n_folds = 7
hrf = 5
srm_k = 30
datadir = '/tigress/jamalw/MES/'
mask_img = load_img(datadir + 'data/mask_nonan.nii')
mask = mask_img.get_data()
mask_reshape = np.reshape(mask,(91*109*91))

results_z = np.zeros((91,109,91,n_folds))
results_real = np.zeros((91,109,91,n_folds))


human_bounds = np.load(datadir + 'prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_wva/' + songs[song_idx] + '/' + songs[song_idx] + '_beh_seg.npy') + hrf

# Custom mean estimator with Fisher z transformation for correlations
def fisher_mean(correlation, axis=None):
    return np.tanh(np.mean(np.arctanh(correlation), axis=axis))

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
                   subj_regs = np.genfromtxt(datadir + subjs[i] + '/EPI_mcf1.par')
                   motion = subj_regs.T
                   regr = linear_model.LinearRegression()
                   regr.fit(motion[:,0:2511].T,subj_data[:,:,0].T)
                   subj_data1 = subj_data[:,:,0] - np.dot(regr.coef_, motion[:,0:2511]) - regr.intercept_[:, np.newaxis]
                   data.append(np.nan_to_num(stats.zscore(subj_data1,axis=1,ddof=1)))
               for i in range(len(subjs)):
                   subj_data = np.load(datadir + subjs[i] + '/' + str(x) + '_' + str(y) + '_' + str(z) + '.npy')
                   subj_regs = np.genfromtxt(datadir + subjs[i] + '/EPI_mcf2.par')
                   motion = subj_regs.T
                   regr = linear_model.LinearRegression()
                   regr.fit(motion[:,0:2511].T,subj_data[:,:,1].T)
                   subj_data2 = subj_data[:,:,1] - np.dot(regr.coef_, motion[:,0:2511]) - regr.intercept_[:, np.newaxis]
                   data.append(np.nan_to_num(stats.zscore(subj_data2,axis=1,ddof=1))) 
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
        print(p)
        vox_z[:,p] = (voxmean[:,p] - fisher_mean(voxmean[:,1:],axis=1))/np.std(voxmean[:,1:],axis=1) 
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
    
    nPerm = 1000
    within_across = np.zeros(nPerm+1)
    run1 = [X[i] for i in np.arange(0, int(len(X)/2))]
    run2 = [X[i] for i in np.arange(int(len(X)/2), len(X))]
    n_iter=10
   
    # run SRM on masked data
    if runNum == 0:
        shared_data = SRM_V1(run2,run1,srm_k,n_iter)
    elif runNum == 1:
        shared_data = SRM_V1(run1,run2,srm_k,n_iter)

    shared_data = stats.zscore(np.dstack(shared_data),axis=1,ddof=1)
    others = np.mean(shared_data[:,song_bounds[song_idx]:song_bounds[song_idx + 1],:13],axis=2)
    loo = np.mean(shared_data[:,song_bounds[song_idx]:song_bounds[song_idx + 1],13:],axis=2) 
    nTR = loo.shape[1]

    # Fit to all but one subject
    K = len(human_bounds) + 1
    ev = brainiak.eventseg.event.EventSegment(K)
    ev.fit(others.T)
    events = np.argmax(ev.segments_[0],axis=1)
    _, event_lengths = np.unique(events, return_counts=True)
    max_event_length = stats.mode(events)[1][0]
 
    # compute timepoint by timepoint correlation matrix 
    cc = np.corrcoef(loo.T) # Should be a time by time correlation matrix

    # Create a mask to only look at values up to max_event_length
    local_mask = np.zeros(cc.shape, dtype=bool)
    for k in range(1,max_event_length):
        local_mask[np.diag(np.ones(cc.shape[0]-k, dtype=bool), k)] = True

    # Compute within vs across boundary correlations, for real and permuted bounds
    for p in range(nPerm+1):
        same_event = events[:,np.newaxis] == events
        within = fisher_mean(cc[same_event*local_mask])
        across = fisher_mean(cc[(~same_event)*local_mask])
        within_across[p] = within - across
            
        np.random.seed(p)
        perm_lengths = np.random.permutation(event_lengths)
        events = np.zeros(nTR, dtype=np.int)
        events[np.cumsum(perm_lengths[:-1])] = 1
        events = np.cumsum(events)

    return within_across


for i in range(n_folds):
    # create coords matrix
    results3d = np.zeros((91,109,91))
    results3d_real = np.zeros((91,109,91))
    results3d_perms = np.zeros((91,109,91,1001))
    results_perms_avg = np.zeros((91,109,91,1001))
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
    results_z[:,:,:,i] = results3d
    results3d_real[mask>0] = raw_wVa_scores[:,0]
    results_real[:,:,:,i] = results3d_real
    if runNum == 0:
        for j in range(vox_z.shape[1]):
            results3d_perms[mask>0,j] = vox_z[:,j]
        np.save('/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_wva_shuffle_event_lengths/' + songs[song_idx] +'/perms/full_brain/globals_perms_test_run1_fisher', results3d_perms)
    elif runNum == 1:
        for j in range(vox_z.shape[1]):
            results3d_perms[mask>0,j] = vox_z[:,j]
        np.save('/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_wva_shuffle_event_lengths/' + songs[song_idx] +'/perms/full_brain/globals_perms_test_run2_fisher', results3d_perms)

results_z    = np.mean(results_z,axis=3)
results_real = fisher_mean(results_real,axis=3) 

# save results 
print('Saving to Searchlight Folders')
if runNum == 0:
    np.save('/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_wva_shuffle_event_lengths/' + songs[song_idx] +'/real/full_brain/globals_K_raw_test_run1_fisher_srm_k' + str(srm_k), results_real)
    np.save('/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_wva_shuffle_event_lengths/' + songs[song_idx] +'/zscores/full_brain/globals_K_zscores_test_run1_fisher_srm_k' + str(srm_k), results_z)
elif runNum == 1:
    np.save('/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_wva_shuffle_event_lengths/' + songs[song_idx] +'/real/full_brain/globals_K_raw_test_run2_fisher_srm_k' + str(srm_k), results_real)
    np.save('/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_wva_shuffle_event_lengths/' + songs[song_idx] +'/zscores/full_brain/globals_K_zscores_test_run2_fisher_srm_k' + str(srm_k), results_z)
