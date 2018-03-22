import numpy as np
import sys
import os
import nibabel as nib
import glob
import scipy.stats as st

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']


# Collect searchlight files
k = '6'

for i in range(len(songs)):
    datadir = '/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/' + songs[i] + '/avg_data/'
    fn = datadir + 'globals_avg_n25_k'+k+'.npy'
    global_outputs_all = np.load(fn)[:,:,:]


    # Reshape data
    z_scores_reshaped = np.nan_to_num(np.reshape(global_outputs_all,(91*109*91)))

    # Mask data with nonzeros
    mask = z_scores_reshaped != 0
    z_scores_reshaped[mask] = st.zscore(z_scores_reshaped[mask])
    z_scores_reshaped[mask] = -np.log(st.norm.sf(z_scores_reshaped[mask]))

    # Reshape data back to original shape
    neg_log_p_values = np.reshape(z_scores_reshaped,(91,109,91))

    # Plot and save searchlight results
    maxval = np.max(neg_log_p_values[~np.isnan(neg_log_p_values)])
    minval = np.min(neg_log_p_values[~np.isnan(neg_log_p_values)])
    img = nib.Nifti1Image(neg_log_p_values, np.eye(4))
    img.header['cal_min'] = minval
    img.header['cal_max'] = maxval
    nib.save(img,datadir + '/globals_avg_n25_k'+k+'_neglog_super_z.nii.gz')


