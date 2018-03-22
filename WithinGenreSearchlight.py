import numpy as np
from nilearn.image import load_img
from brainiak.searchlight.searchlight import Searchlight
from scipy import stats
import nibabel as nib

# Take subject ID as input
subjs = ['MES_022817_0','MES_030217_0','MES_032117_1','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0','MES_041417_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

datadir = '/tigress/jamalw/MES/'
mask_img = load_img(datadir + 'data/mask_nonan.nii.gz')
mask_img = mask_img.get_data()
global_outputs_all = np.zeros((91,109,91,len(subjs)))


# Definte function that takes the difference between within vs. between genre comparisons
def corr2_coeff(AB,msk,myrad,bcast_var):
    #if not np.all(msk):
    #    return None
    A,B = (AB[0], AB[1])
    A = A[msk==1,:]
    B = B[msk==1,:]
    A = A.reshape((-1,A.shape[-1]))
    B = B.reshape((-1,B.shape[-1]))
    corr_eye = np.identity(8)
    corrAB = np.corrcoef(A.T,B.T)[16:,:16]
    jazz_within  = corrAB[8:16,8:16]
    JazzWithinAvgOn = np.mean(jazz_within[corr_eye == 1])
    JazzBtwnAvgOff = np.mean(jazz_within[corr_eye == 0])
    classical_within  = corrAB[0:8,0:8]
    ClassicalWithinAvgOn = np.mean(classical_within[corr_eye == 1])
    ClassicalBtwnAvgOff = np.mean(classical_within[corr_eye == 0]) 
    jazz_diff = JazzWithinAvgOn - JazzBtwnAvgOff
    classical_diff = ClassicalWithinAvgOn - ClassicalBtwnAvgOff
    diff = (jazz_diff + classical_diff)/2

    # compute difference score for permuted matrices    
    np.random.seed(0)
    diff_perm_holder = []
    for i in range(1000): 
        jazz_within_perm = jazz_within[np.random.permutation(8),:]
        classical_within_perm = classical_within[np.random.permutation(8),:]
        JazzWithinAvgOn_perm = np.mean(jazz_within_perm[corr_eye == 1])
        ClassicalWithinAvgOn_perm = np.mean(classical_within_perm[corr_eye == 1])
        JazzBtwnAvgOff_perm = np.mean(jazz_within_perm[corr_eye == 0])
        ClassicalBtwnAvgOff_perm = np.mean(classical_within_perm[corr_eye == 0]) 
        jazz_diff_perm = JazzWithinAvgOn_perm - JazzBtwnAvgOff_perm
        classical_diff_perm = ClassicalWithinAvgOn_perm - ClassicalBtwnAvgOff_perm
        diff_perm = (jazz_diff_perm + classical_diff_perm)/2
        diff_perm_holder.append(diff_perm)

    z = (diff - np.mean(diff_perm_holder))/np.std(diff_perm_holder)
    print(z)
    return z

for i in range(len(subjs)):
    # Load functional data and mask data
    data1 = load_img(datadir + 'subjects/' + subjs[i] + '/data/avg_reorder1.nii')
    data2 = load_img(datadir + 'subjects/' + subjs[i] + '/data/avg_reorder2.nii')
    data1 = data1.get_data()
    data2 = data2.get_data()
    print('Subject: ', subjs[i])

    # Create and run searchlight
    sl = Searchlight(sl_rad=5,max_blk_edge=5)
    sl.distribute([data1,data2],mask_img)
    sl.broadcast(None)
    print('Running Searchlight...')
    global_outputs = sl.run_searchlight(corr2_coeff)
    global_outputs_all[:,:,:,i] = global_outputs    
 
# Plot and save searchlight results
savedir = datadir + 'prototype/link/scripts/data/searchlight_output/Within_Genre_Results/'  
global_outputs_avg = np.mean(global_outputs_all,3)
global_outputs = np.array(global_outputs_avg, dtype=np.float)
global_nonans = global_outputs[np.not_equal(global_outputs,None)]
global_nonans = np.reshape(global_nonans,(91,109,91))
min1 = np.min(global_nonans[~np.isnan(global_nonans)])
max1 = np.max(global_nonans[~np.isnan(global_nonans)])
img = nib.Nifti1Image(global_nonans,np.eye(4))
img.header['cal_min'] = min1
img.header['cal_max'] = max1
nib.save(img, savedir + 'within_genre.nii.gz')
np.save(savedir + 'within_genre', global_nonans)




