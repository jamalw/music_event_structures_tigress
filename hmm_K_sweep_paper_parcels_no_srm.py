from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import numpy as np
import brainiak.eventseg.event
from scipy.stats import norm, zscore, pearsonr, stats
from scipy.signal import gaussian, convolve
from sklearn import decomposition
import numpy as np
from brainiak.funcalign.srm import SRM
import sys


# Custom mean estimator with Fisher z transformation for correlations
def fisher_mean(correlation, axis=None):
    return np.tanh(np.mean(np.arctanh(correlation), axis=axis))

datadir = '/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_input/Schaefer300/'
ann_dirs = '/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'
savedir = '/tigress/jamalw/MES/prototype/link/scripts/hmm_K_sweep_paper_results/Schaefer300/DMN_no_srm/'

numParcels = 300

bootNum = int(sys.argv[1])
roiNum = int(sys.argv[2])

nboot = 50

K_set = np.array((3,5,9,15,20,25,30,35,40,45))

#########################################################################################
# Here we train and test the model on both runs separately. This will result in two SRM-ified datasets: one for all of run 1 and one for all of run 2. Songs from these datasets will be indexed separately in the following HMM step and then averaged before fitting the HMM.

# run 1 times
song_bounds_run1 = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973, 2198,2377,2511])

songs_run1 = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

durs_run1 = np.array([225,90,180,135,90,180,135,90,180,135,225,90,225,225,180,135])

# run 2 times
song_bounds_run2 = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

songs_run2 = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

durs_run2 = np.array([90,180,180,90,135,180,180,225,225,135,90,135,225,225,90,135])


# Load in data and reshape for Schaefer parcellations where the dimensionality is nSubjs X nVox X Time whereas the dimensions for the data used in the original version of the analysis was nVox X Time X nSubjs 
run1 = np.load(datadir + 'parcel' + str(roiNum) + '_run1.npy')
run2 = np.load(datadir + 'parcel' + str(roiNum) + '_run2.npy')

nSubj = run1.shape[0]

wVa_results = np.zeros((16,len(K_set),nboot))

np.random.seed(bootNum)

for b in range(nboot):
        resamp_subjs = np.random.choice(nSubj, size=nSubj, replace=True)
        run1_resample = run1[resamp_subjs,:,:]
        run2_resample = run2[resamp_subjs,:,:]

        run1_resample_avg = np.mean(run1_resample,axis=0)
        run2_resample_avg = np.mean(run2_resample,axis=0)


	##################################################################################

        for i in range(16):
            print('song number ',str(i))
            # grab start and end time for each song from bound vectors. for SRM data trained on run 1 and tested on run 2, use song name from run 1 to find index for song onset in run 2 bound vector 
            start_run1 = song_bounds_run1[i]
            end_run1   = song_bounds_run1[i+1] 
            start_run2 = song_bounds_run2[songs_run2.index(songs_run1[i])]
            end_run2  = song_bounds_run2[songs_run2.index(songs_run1[i])+1]
            # chop song from bold data
            data1 = run1_resample_avg[:,start_run1:end_run1]
            data2 = run2_resample_avg[:,start_run2:end_run2]
            # average song-specific bold data from each run 
            data = (data1 + data2)/2
            for j in range(len(K_set)):
                # Fit HMM
                ev = brainiak.eventseg.event.EventSegment(int(K_set[j]))
                ev.fit(data.T)
                events = np.argmax(ev.segments_[0],axis=1)
                                 
                max_event_length = stats.mode(events)[1][0]
                # compute timepoint by timepoint correlation matrix 
                cc = np.corrcoef(data.T) # Should be a time by time correlation matrix
                         
                # Create a mask to only look at values up to max_event_length
                local_mask = np.zeros(cc.shape, dtype=bool)
                for k in range(1,max_event_length):
                    local_mask[np.diag(np.ones(cc.shape[0]-k, dtype=bool), k)] = True
                      
                    # Compute within vs across boundary correlations
                    same_event = events[:,np.newaxis] == events
                    within = fisher_mean(cc[same_event*local_mask])
                    across = fisher_mean(cc[(~same_event)*local_mask])
                    within_across = within - across
                    wVa_results[i,j,b] = within_across


np.save(savedir + 'parcel' + str(roiNum) + '_' + str(bootNum), wVa_results)
