import numpy as np
import soundfile as sf
import nibabel as nib

MES_dir = '/jukebox/norman/jamalw/MES/'

subjs = ['MES_022817_0','MES_030217_0', 'MES_032117_1','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0', 'MES_041417_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

songs = ['Island','Capriccio_Espagnole','Waltz_of_the_Flowers','St.Pauls_Suite','Moonlight_Sonata','Symphonie_Fantastique','Allegro_Moderato','Finlandia','My_Favorite_Things','All_Blues','Boogie_Stop_Shuffle','The_Bird','Blue_Monk','Change_of_the_Guard','Early_Summer','I_Love_Music']

subj_format = 'subj%d'

print("Loading song duration data...")
# load song data
songs1Dur = np.load(MES_dir + 'data/' + 'songs1Dur.npy')
songs2Dur = np.load(MES_dir + 'data/' + 'songs2Dur.npy')

for i in range(0,len(subjs)):
    run1 = nib.load(MES_dir + 'subjects/' + subjs[i] + '/analysis/run1.feat/trans_filtered_func_data.nii').get_data()
    run2 = nib.load(MES_dir + 'subjects/' + subjs[i] + '/analysis/run2.feat/trans_filtered_func_data.nii').get_data()
    print("Slicing functional data by song durations...")
    # slice functional scan according to song durations
    func_sliced1 = []
    func_sliced2 = []
    for j in range(0,len(songs1Dur)):
        func_sliced1.append([])
        func_sliced2.append([])

    for j in range(0,len(songs1Dur)):
        func_sliced1[j].append(run1[:,:,:,0:songs1Dur[j]])
        func_sliced2[j].append(run2[:,:,:,0:songs2Dur[j]])
        run1 = run1[:,:,:,songs1Dur[j]:]
        run2 = run2[:,:,:,songs2Dur[j]:]    
    # create subject general song model for both experiments
    exp1 = np.array([7, 12, 15,  2,  1,  0,  9,  3,  4,  5,  6, 13, 10,  8, 11, 14])
    exp2 = np.array([3, 15,  4, 13,  2, 11,  0,  6,  7, 14,  1,  5, 10,  8, 12, 9])

    print("Reordering functional data to match genre model...")
    # reorder func data according to genre model
    reorder1 = []
    reorder2 = []
    reorder_avg = []
    # create 16 slots
    for j in range(0,len(exp1)):
        reorder1.append([])
        reorder2.append([])
        reorder_avg.append([])
    # assign successive songs in func_slicedn to respective index in reordern.
    for j in range(0,len(exp1)):
        reorder1[exp1[j]] = func_sliced1[j][0]
        reorder2[exp2[j]] = func_sliced2[j][0]
    # average same songs together and save to respective directory
    for j in range(0,len(exp1)):
        reorder_avg[j] = (reorder1[j]+reorder2[j])/2
        
    for j in range(0,len(exp1)):
        min1 = np.min(reorder_avg[j])
        max1 = np.max(reorder_avg[j])
        img1 = nib.Nifti1Image(reorder_avg[j],np.eye(4))
        img1.header['cal_min'] = min1
        img1.header['cal_max'] = max1
        nib.save(img1,MES_dir + 'data/single_song_niftis/' + songs[j] + '/' + subj_format % (i+1) + '.nii.gz')        


 
