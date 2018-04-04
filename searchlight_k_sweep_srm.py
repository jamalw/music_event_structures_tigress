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

k_sweeper = [5]
loo_idx = int(sys.argv[1])
song_idx = int(sys.argv[2])
subj = subjs[int(loo_idx)]
print('Subj: ', subj)

datadir = '/tigress/jamalw/MES/'
mask_img = load_img('/tigress/jamalw/MES/data/a1plus_2mm.nii.gz')
mask = mask_img.get_data()
mask_reshape = np.reshape(mask,(91*109*91))

def searchlight(coords,K,mask,loo_idx,subjs,song_idx,song_bounds):
    
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
    min_vox = 10
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
               SL_within_across = HMM(data,K,loo_idx,song_idx,song_bounds)
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

def HMM(X,K,loo_idx,song_idx,song_bounds):
    
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
    
    w = 6
    srm_k = 45
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
    others = np.mean(shared_data[:,:,np.arange(shared_data.shape[-1]) != loo_idx],axis=2)
    loo = shared_data[:,song_bounds[song_idx]:song_bounds[song_idx + 1],loo_idx] 
    nTR = loo.shape[1]

    # Fit to all but one subject
    ev = brainiak.eventseg.event.EventSegment(K)
    ev.fit(others[:,song_bounds[song_idx]:song_bounds[song_idx + 1]].T)
    events = np.argmax(ev.segments_[0],axis=1)

    ####
    # plot searchlights
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    shared_data = srm.transform(run2)
    avg_response = sum(shared_data)/len(shared_data)
    plt.figure(figsize=(10,10))
    plt.imshow(np.corrcoef(avg_response[:,0:89].T))
    bounds = np.where(np.diff(np.argmax(ev.segments_[0], axis=1)))[0]
    ax = plt.gca()
    bounds_aug = np.concatenate(([0],bounds,[nTR]))
    for i in range(len(bounds_aug)-1):
        rect1 = patches.Rectangle((bounds_aug[i],bounds_aug[i]),bounds_aug[i+1]-bounds_aug[i],bounds_aug[i+1]-bounds_aug[i],linewidth=3,edgecolor='w',facecolor='none',label='Model Fit')
        ax.add_patch(rect1)
    plt.title('HMM Fit to A1 SRM K = ' + str(srm_k),fontsize=18,fontweight='bold')
    plt.savefig('plots/St_Pauls SRM K = ' + str(srm_k))
    ####

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

    return within_across


for i in k_sweeper:
    # create coords matrix
    global_outputs_all = np.zeros((91,109,91))
    results3d = np.zeros((91,109,91))
    results3d_real = np.zeros((91,109,91))
    results3d_perms = np.zeros((91,109,91,1001))
    x,y,z = np.mgrid[[slice(dm) for dm in tuple((91,109,91))]]
    x = np.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2]))
    y = np.reshape(y,(y.shape[0]*y.shape[1]*y.shape[2]))
    z = np.reshape(z,(z.shape[0]*z.shape[1]*z.shape[2]))
    coords = np.vstack((x,y,z)).T 
    coords_mask = coords[mask_reshape>0]
    print('Running Distribute...')
    vox_z,raw_wVa_scores = searchlight(coords_mask,i,mask,loo_idx,subjs,song_idx,song_bounds) 
    results3d[mask>0] = vox_z[:,0]
    results3d_real[mask>0] = raw_wVa_scores[:,0]
    for j in range(vox_z.shape[1]):
        results3d_perms[mask>0,j] = vox_z[:,j]
 
    print('Saving ' + subj + ' to Searchlight Folder')
    np.save('/scratch/gpfs/jamalw/HMM_searchlight_K_sweep_srm/' + songs[song_idx] +'/real/globals_loo_' + subj + '_K_real' + str(i), results3d_real)
    np.save('/scratch/gpfs/jamalw/HMM_searchlight_K_sweep_srm/' + songs[song_idx] +'/zscores/globals_loo_' + subj + '_K_' + str(i), results3d)
    np.save('/scratch/gpfs/jamalw/HMM_searchlight_K_sweep_srm/' + songs[song_idx] +'/perms/globals_loo_' + subj + '_K_perms' + str(i), results3d_perms)


