import numpy as np
import nibabel as nib
from scipy import stats

nii_template = nib.load('/tigress/jamalw/MES/data/trans_filtered_func_data.nii')
datadir = '/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'

# load,zscore,then store each dataset for each K in a list
r_data = np.zeros((91,109,91,7))
z_data = np.zeros((91,109,91,7))
thresh = np.zeros((91,109,91))
maxval_per_maxK = np.zeros((91,109,91,7)) 
i_array = np.arange(3,16,2)

for counter, i in enumerate(i_array):
    data = nib.load(datadir + 'avg_real_k' + str(i) + '_across_songs.nii.gz').get_data()
    data[np.isnan(data)] = 0
    z = nib.load(datadir + 'avg_z_k' + str(i) + '_across_songs.nii.gz').get_data()
    z[np.isnan(z)] = 0
    r_data[:,:,:,counter] = data
    z_data[:,:,:,counter] = z

max_data = np.max(r_data,axis=3)
max_K = np.argmax(r_data,axis=3)
max_K[np.sum(r_data, axis=3) == 0] = 0       
 
for i in range(91):
    for j in range(109):
        for k in range(91):
            thresh[i,j,k] = z_data[i,j,k,max_K[i,j,k]]

max_K = i_array[max_K]
max_K[thresh < 0] = 0
max_K = max_K.astype(float)

# save final map as nifti
maxval = np.max(max_K)
minval = np.min(max_K)
img = nib.Nifti1Image(max_K,affine = nii_template.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img, datadir + 'best_k_map.nii.gz')


