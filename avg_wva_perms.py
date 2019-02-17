import numpy as np
import glob

datadir = "/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_wva/"

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

for s in range(len(songs)):
    avg_data = np.zeros((91,109,91,1001))
    fn = glob.glob(datadir + songs[s] + '/perms/full_brain/*run1*')
    for i in range(len(fn)):
        data = np.load(fn[i])
        avg_data[:,:,:,:] += data/len(fn)
    np.save(datadir + songs[s] + '/perms/full_brain/avg_perms_train_run1',avg_data) 

for s in range(len(songs)):
    avg_data = np.zeros((91,109,91,1001))
    fn = glob.glob(datadir + songs[s] + '/perms/full_brain/*run2*')
    for i in range(len(fn)):
        data = np.load(fn[i])
        avg_data[:,:,:,:] += data/len(fn)
    np.save(datadir + songs[s] + '/perms/full_brain/avg_perms_train_run2',avg_data)

for s in range(len(songs)):
    run1 = np.load(datatdir + songs[s] + '/perms/full_brain/avg_perms_train_run1.npy')
    run2 = np.load(datadir + songs[s] + '/perms/full_brain/avg_perms_train_run2.npy')
    avg_runs = (run1 + run2)/2
    np.save(datadir + songs[s] + '/perms/full_brain/avg_perms_both_runs',avg_runs)
