import numpy as np
import nibabel as nib

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard', 'Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk', 'All_Blues']

#k = np.arange(2,20)
k = np.array([3,5,7,9,11,13,15])
nii_template = nib.load('/tigress/jamalw/MES/data/trans_filtered_func_data.nii')

## Z-SCORED SEARCHLIGHT RESULTS

# Take average K across songs
for i in range(len(k)):
    avg_data = np.zeros((91,109,91))
    for j in range(len(songs)):
        datadir = '/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/' + songs[j] + '/zscores/full_brain/' 
        fn = datadir + 'globals_K_zscores_' + str(k[i]) + '.npy'
        data = np.load(fn)
        avg_data[:,:,:] += data/(len(songs))
    maxval = np.max(avg_data[~np.isnan(avg_data)])
    minval = np.min(avg_data[~np.isnan(avg_data)])
    img = nib.Nifti1Image(avg_data, affine=nii_template.affine)
    img.header['cal_min'] = minval
    img.header['cal_max'] = maxval
    nib.save(img,'/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/avg_z_k' + str(k[i]) +'_across_songs.nii.gz')


## RAW SEARCHLIGHT RESULTS

# Take average K across songs
for i in range(len(k)):
    avg_data = np.zeros((91,109,91))
    for j in range(len(songs)):
        datadir = '/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/' + songs[j] + '/real/full_brain/' 
        fn = datadir + 'globals_K_raw_' + str(k[i]) + '.npy'
        data = np.load(fn)
        avg_data[:,:,:] += data/(len(songs))
    maxval = np.max(avg_data[~np.isnan(avg_data)])
    minval = np.min(avg_data[~np.isnan(avg_data)])
    img = nib.Nifti1Image(avg_data, affine=nii_template.affine)
    img.header['cal_min'] = minval
    img.header['cal_max'] = maxval
    nib.save(img,'/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/avg_real_k' + str(k[i]) +'_across_songs.nii.gz')


