import numpy as np
import nibabel as nib
from scipy import stats

nii_template = nib.load('/tigress/jamalw/MES/subjects/MES_022817_0/analysis/run1.feat/trans_filtered_func_data.nii')
datadir = '/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'

# load,zscore,then store each dataset for each K in a list
k_data = np.zeros((91,109,91,10))
z_data = np.zeros((91,109,91,10))
thresh = np.zeros((91,109,91))
maxval_per_maxK = np.zeros((91,109,91,10)) 

for i in range(3,13):
    data = nib.load(datadir + 'avg_real_k' + str(i) + '_across_songs.nii.gz').get_data()
    z = nib.load(datadir + 'avg_z_k' + str(i) + '_across_songs.nii.gz').get_data()
    k_data[:,:,:,i-3] = data
    z_data[:,:,:,i-3] = z 

max_data = np.max(k_data,axis=3)
max_K = np.argmax(k_data,axis=3) + 3
max_K[np.sum(k_data, axis=3) == 0] = 0       
 
for i in range(91):
    for j in range(109):
        for k in range(91):
            thresh[i,j,k] = z_data[i,j,k,max_K[i,j,k]-3]

max_K[thresh < 0] = 0
#max_K_mask = max_K == 3
#max_K = max_K_mask * 3
#max_K = np.where(max_K == 3,3,0)
#max_K4 = np.where(max_K == 4,4,0)
k = 3
z_thresh = 0.02
output = np.where((max_K == k) & (z_data[:, :, :, k-1] >= z_thresh), k, 0)

#stack = []
#for k in range(3, 13):
#     stack.append(np.where((max_K == k) & (z_data[:, :, :, k-3] > z_thresh), k, 0))
#all_max_K = np.sum(stack, axis=0)

max_K = max_K.astype(float)
# save final map as nifti
maxval = np.max(max_K)
minval = np.min(max_K)
img = nib.Nifti1Image(max_K,affine = nii_template.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img, datadir + 'best_k_map.nii.gz')

maxval = np.max(output)
minval = np.min(output)
img = nib.Nifti1Image(output,affine = nii_template.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img, datadir + 'single_k_map.nii.gz')

