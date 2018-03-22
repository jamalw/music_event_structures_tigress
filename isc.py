import numpy as np
import nibabel as nib
import numpy.ma as ma
# This script calls the brainiak isc and isfc function to perform full brain isc on music data.


#subjs = ['MES_022817_0','MES_030217_0','MES_032117_1','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0', 'MES_041417_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

subjs = ['MES_022817_0','MES_030217_0','MES_032117_1']

datadir = '/jukebox/norman/jamalw/MES/'

# Load mask
mask_fn = 'mask_nonan.nii.gz'
mask = nib.load(datadir + 'data/' + mask_fn).get_data()

# Compute ISC
for i in range(len(subjs)):
    print('Loading Subject: ',subjs[i])
    loo = nib.load(datadir + 'subjects/' + subjs[i] + '/analysis/run1.feat/trans_filtered_func_data.nii').get_data()
    loo = loo[mask == 1] 
    others = np.load(datadir + 'prototype/link/scripts/data/avg_others_run1/loo_' + str(i) + '.npy')
    others = others[mask == 1,-1]
 

print('Collected Subject Data')



save_dir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/'
print('Saving ISC Results')
np.save(save_dir + 'full_brain_ISC_run1',ISC1)
np.save(save_dir + 'full_brain_ISC_run2',ISC2)
