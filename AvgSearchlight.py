import numpy as np
import sys
import os
import nibabel as nib
import glob
import scipy.stats as st

datadir = '/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/'
searchlight_dir = sys.argv[1]
datadir_full = datadir + searchlight_dir + '/'

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

# Collect searchlight files
k = ['6']

for s in range(len(songs)):
    for i in range(len(k)):
        fn = glob.glob(datadir_full + songs[s] + '/' + '*_'+k[i]+'.npy')

        global_outputs_all = np.zeros((91,109,91))

        # Take average of searchlight results
        for j in range(0,len(fn)):
            subj_data = np.load(fn[j])
            global_outputs_all[:,:,:] += subj_data/(len(fn)) 
    
        # Save average results 
        np.save(datadir_full + songs[s] + '/avg_data/globals_avg_n25_k'+k[i],global_outputs_all)

        # Reshape data
        z_scores_reshaped = np.nan_to_num(np.reshape(global_outputs_all,(91*109*91)))
        z_scores_reshaped_pval = np.nan_to_num(np.reshape(global_outputs_all,(91*109*91)))

        # Mask data with nonzeros
        mask = z_scores_reshaped != 0
        z_scores_reshaped[mask] = -np.log(st.norm.sf(z_scores_reshaped[mask]))
        z_scores_reshaped_pval[mask] = st.norm.sf(z_scores_reshaped_pval[mask])

        # Reshape data back to original shape
        neg_log_p_values = np.reshape(z_scores_reshaped,(91,109,91))
        p_values = np.reshape(z_scores_reshaped_pval,(91,109,91))

        # Plot and save searchlight results
        maxval = np.max(neg_log_p_values[~np.isnan(neg_log_p_values)])
        minval = np.min(neg_log_p_values[~np.isnan(neg_log_p_values)])
        img = nib.Nifti1Image(neg_log_p_values, np.eye(4))
        img.header['cal_min'] = minval
        img.header['cal_max'] = maxval
        nib.save(img,datadir_full + songs[s] + '/avg_data/globals_avg_n25_k'+k[i]+'_neglog.nii.gz')

        maxval2 = np.max(p_values[~np.isnan(p_values)])
        minval2 = np.min(p_values[~np.isnan(p_values)])
        img2 = nib.Nifti1Image(p_values, np.eye(4))
        img2.header['cal_min'] = minval2
        img2.header['cal_max'] = maxval2
        nib.save(img2,datadir_full + songs[s] + '/avg_data/globals_avg_n25_k'+k[i]+'_pvals.nii.gz') 
