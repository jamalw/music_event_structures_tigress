import numpy as np
import nibabel as nib

songs = ['I_Love_Music', 'Moonlight_Sonata', 'Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'All_Blues']

k = np.arange(2,20)
nii_template = nib.load('/tigress/jamalw/MES/data/trans_filtered_func_data.nii')

## Z-SCORED SEARCHLIGHT RESULTS

# Take average K within songs. This should tell you which voxels were most active for a particular song regardless of K.
for i in range(len(songs)):
    avg_data = np.zeros((91,109,91))
    datadir = '/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/' + songs[i] + '/avg_data/'
    for j in range(len(k)):
        fn = datadir + 'globals_avg_z_n25_k' + str(k[j]) + '.npy'
        subj_data = np.load(fn)
        avg_data[:,:,:] += subj_data/(len(k))
    maxval = np.max(avg_data[~np.isnan(avg_data)])
    minval = np.min(avg_data[~np.isnan(avg_data)])
    img = nib.Nifti1Image(avg_data, affine=nii_template.affine)
    img.header['cal_min'] = minval
    img.header['cal_max'] = maxval
    nib.save(img,datadir + 'avg_z_across_K.nii.gz')


# Take average across songs
for i in range(len(k)):
    avg_data = np.zeros((91,109,91))
    for j in range(len(songs)):
        datadir = '/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/' + songs[j] + '/avg_data/' 
        fn = datadir + 'globals_avg_z_n25_k' + str(k[i]) + '.npy'
        subj_data = np.load(fn)
        avg_data[:,:,:] += subj_data/(len(songs))
    maxval = np.max(avg_data[~np.isnan(avg_data)])
    minval = np.min(avg_data[~np.isnan(avg_data)])
    img = nib.Nifti1Image(avg_data, affine=nii_template.affine)
    img.header['cal_min'] = minval
    img.header['cal_max'] = maxval
    nib.save(img,'/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/avg_z_k' + str(k[i]) +'_across_songs.nii.gz')


## RAW SEARCHLIGHT RESULTS

# Take average K within songs. This should tell you which voxels were most active for a particular song regardless of K.
for i in range(len(songs)):
    avg_data = np.zeros((91,109,91))
    datadir = '/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/' + songs[i] + '/avg_data/'
    for j in range(len(k)):
        fn = datadir + 'globals_avg_real_n25_k' + str(k[j]) + '.npy'
        subj_data = np.load(fn)
        avg_data[:,:,:] += subj_data/(len(k))
    maxval = np.max(avg_data[~np.isnan(avg_data)])
    minval = np.min(avg_data[~np.isnan(avg_data)])
    img = nib.Nifti1Image(avg_data, affine=nii_template.affine)
    img.header['cal_min'] = minval
    img.header['cal_max'] = maxval
    nib.save(img,datadir + 'avg_real_across_K.nii.gz')


# Take average across songs
for i in range(len(k)):
    avg_data = np.zeros((91,109,91))
    for j in range(len(songs)):
        datadir = '/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/' + songs[j] + '/avg_data/' 
        fn = datadir + 'globals_avg_real_n25_k' + str(k[i]) + '.npy'
        subj_data = np.load(fn)
        avg_data[:,:,:] += subj_data/(len(songs))
    maxval = np.max(avg_data[~np.isnan(avg_data)])
    minval = np.min(avg_data[~np.isnan(avg_data)])
    img = nib.Nifti1Image(avg_data, affine=nii_template.affine)
    img.header['cal_min'] = minval
    img.header['cal_max'] = maxval
    nib.save(img,'/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/avg_real_k' + str(k[i]) +'_across_songs.nii.gz')


