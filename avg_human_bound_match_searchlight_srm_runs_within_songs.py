import numpy as np
import glob
import nibabel as nib


songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

nii_template = nib.load('/jukebox/norman/jamalw/MES/subjects/MES_022817_0/analysis/run1.feat/trans_filtered_func_data.nii')


for i in range(len(songs)):
    datadir = '/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/music_features/' + songs[i] 
    fn_z_run1   = np.load(datadir + '/zscores/globals_z_srm_k30_fit_run1_psVox.npy')
    fn_z_run2   = np.load(datadir + '/zscores/globals_z_srm_k30_fit_run2_psVox.npy')
    
    avg_both_z_runs = (z_data_run1 + z_data_run2)/2   

    # Save all zscore averages
    maxval = np.max(avg_both_z_runs)
    minval = np.min(avg_both_z_runs)
    img = nib.Nifti1Image(avg_both_z_runs, affine=nii_template.affine)
    img.header['cal_min'] = minval
    img.header['cal_max'] = maxval
    nib.save(img,datadir + '/zscores/globals_avg_both_z_runs_psVox.nii.gz')  
    

 
