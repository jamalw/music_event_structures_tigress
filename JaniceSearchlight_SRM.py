# This script computes the pattern similarity between subjects at every voxel in the brain via a Brainiak searchlight.

# Author: Jamal Williams
# Princeton Neuroscience Institute, Princeton University 2017

import numpy as np
from nilearn.image import load_img
import sys
from brainiak.searchlight.searchlight import Searchlight
from scipy import stats
import numpy as np
import nibabel as nib
from brainiak.funcalign.srm import SRM
import numpy.ma as ma

# Take subject ID as input
subjs = ['MES_022817_0','MES_030217_0','MES_032117_1','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0','MES_041417_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

#subjs = ['MES_022817_0', 'MES_030217_0','MES_032117_1']
 
datadir = '/tigress/jamalw/MES/'
mask_img = load_img(datadir + 'data/mask_nonan.nii.gz').get_data()
global_outputs_all = np.zeros((91,109,91,len(subjs)))
niter = 10
nfeature = 10
loo_idx = int(sys.argv[1])

def corr2_coeff(A,msk,myrad,bcast_var):
    #if not np.all(msk):
    #    return None
    print('Assigning Masked Data')
    run1 = [A[i][msk==1] for i in np.arange(0, int(len(A)/2))]
    run2 = [A[i][msk==1] for i in np.arange(int(len(A)/2), len(A))]
    print('Building Model')
    srm = SRM(bcast_var[0],bcast_var[1])
    print('Training Model')
    srm.fit(run1)
    print('Testing Model')
    shared_data = srm.transform(run2)
    shared_data = stats.zscore(np.dstack(shared_data),axis=1,ddof=1)
    others = np.mean(shared_data[:,:,np.arange(shared_data.shape[-1]) != loo_idx],axis=2) 
    loo    = shared_data[:,:,loo_idx] 
    corrAB = np.corrcoef(loo.T,others.T)[16:,:16]
    corr_eye = np.identity(16)
    same_songs = corrAB[corr_eye == 1]
    diff_songs = corrAB[corr_eye == 0]	    
    avg_same_songs = np.mean(same_songs)
    avg_diff_songs = np.mean(diff_songs)
    same_song_minus_diff_song = avg_same_songs - avg_diff_songs
    # Compute difference score for permuted matrices
    np.random.seed(0)
    diff_perm_holder = np.zeros((1000,1))
    for i in range(1000):
        corrAB_perm = corrAB[np.random.permutation(16),:]
        same_songs_perm = corrAB_perm[corr_eye == 1]
        diff_songs_perm = corrAB_perm[corr_eye == 0]
        diff_perm_holder[i] = np.mean(same_songs_perm) - np.mean(diff_songs_perm)                
             
    z = (same_song_minus_diff_song - np.mean(diff_perm_holder))/np.std(diff_perm_holder)
    return z


runs = []


for i in range(len(subjs)):
    # Load functional data and mask data
    print('Subject: ',subjs[i])
    data_run1 = np.nan_to_num(load_img(datadir + 'subjects/' + subjs[i] + '/analysis/run1.feat/trans_filtered_func_data.nii').get_data()[:,:,:,0:628])
    runs.append(data_run1)
for i in range(len(subjs)):
    data_run2 = np.nan_to_num(load_img(datadir + 'subjects/' + subjs[i] + '/data/avg_reorder2.nii').get_data())
    runs.append(data_run2)
    
print("All Subjects Loaded")
         
            
#np.seterr(divide='ignore',invalid='ignore')

# Create and run searchlight
sl = Searchlight(sl_rad=5,max_blk_edge=5)
sl.distribute(runs,mask_img)
sl.broadcast([nfeature,niter,loo_idx])
print('Running Searchlight...')
global_outputs = sl.run_searchlight(corr2_coeff)
global_outputs_all[:,:,:,i] = global_outputs
        
# Plot and save searchlight results
global_outputs_avg = np.mean(global_outputs_all,3)
#maxval = np.max(global_outputs_avg[np.not_equal(global_outputs_avg,None)])
#minval = np.min(global_outputs_avg[np.not_equal(global_outputs_avg,None)])
global_outputs_avg = np.array(global_outputs_avg, dtype=np.float)
#global_nonans = global_outputs_avg[np.not_equal(global_outputs_avg,None)]
#global_nonans = np.reshape(global_nonans,(91,109,91))
#img = nib.Nifti1Image(global_nonans, np.eye(4))
#img.header['cal_min'] = minval
#img.header['cal_max'] = maxval
#nib.save(img,datadir + 'prototype/link/scripts/data/searchlight_output/janice_srm_results/loo_' + subjs[loo_idx])
np.save(datadir + 'prototype/link/scripts/data/searchlight_output/janice_srm_results/loo_' + subjs[loo_idx],global_outputs_avg)

print('Searchlight is Complete!')

