import numpy as np
import nibabel as nib

datadir = "/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/"
nii_template = nib.load('/tigress/jamalw/MES/data/trans_filtered_func_data.nii')

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

hmmK = ['11']

for i in range(0,len(songs)):
    # set data directory for each song
    r_datadir = datadir + songs[i] + '/real/full_brain/'
    z_datadir = datadir + songs[i] + '/zscores/full_brain/'
    
    # load in data
    rdata = np.load(r_datadir + '/globals_K_raw_' + hmmK[0] + '.npy')
    zdata = np.load(z_datadir + '/globals_K_zscores_' + hmmK[0] + '.npy') 
    
    rdata[np.isnan(rdata)] = 0
    zdata[np.isnan(zdata)] = 0

    # convert raw .npy files to .nii
    r_maxval = np.max(rdata)
    r_minval = np.min(rdata)
    r_img = nib.Nifti1Image(rdata, affine=nii_template.affine)
    r_img.header['cal_min'] = r_minval
    r_img.header['cal_max'] = r_maxval
    nib.save(r_img, r_datadir + 'globals_raw_K' + hmmK[0] + '.nii.gz') 

    # convert zscore .npy files to .nii
    z_maxval = np.max(zdata)
    z_minval = np.min(zdata)
    z_img = nib.Nifti1Image(zdata, affine=nii_template.affine)
    z_img.header['cal_min'] = z_minval
    z_img.header['cal_max'] = z_maxval
    nib.save(z_img, z_datadir + 'globals_zscores_K' + hmmK[0] + '.nii.gz') 



