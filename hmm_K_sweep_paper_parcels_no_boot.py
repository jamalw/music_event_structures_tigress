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
from scipy.ndimage.filters import gaussian_filter1d
from statsmodels.nonparametric.kernel_regression import KernelReg

roiNum = str(sys.argv[1])

# Custom mean estimator with Fisher z transformation for correlations
def fisher_mean(correlation, axis=None):
    return np.tanh(np.mean(np.arctanh(correlation), axis=axis))

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_input/parcels/Schaefer300/'
savedir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/hmm_K_sweep_paper_results/Schaefer300/allROIs/'

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

dict_names = ['WvA_Songs_x_Ks','Smooth_WvA','Smooth_Max_WvA','Raw_Max_WvA','Pref_Event_Length_Sec']

# Load in data
run1 = np.load(datadir + 'parcel' + roiNum + '_run1.npy')
run2 = np.load(datadir + 'parcel' + roiNum + '_run2.npy')

nSubj = run1.shape[0]

# Convert data into lists where each element is a subject's data of size voxels by samples
run1_list = []
run2_list = []
for i in range(0,nSubj):
    run1_list.append(run1[i,:,:])
    run2_list.append(run2[i,:,:])

run1_list = run1_list.copy()
run2_list = run2_list.copy()

ROI_WvA = np.zeros((16,len(K_set)))

n_iter = 50
features = 10
# Initialize model
print('Building Model')
srm_train_run1 = SRM(n_iter=n_iter, features=features)
srm_train_run2 = SRM(n_iter=n_iter, features=features)

# Fit model to training data
print('Training Model')
srm_train_run1.fit(run1_list)
srm_train_run2.fit(run2_list)

# Test model on testing data to produce shared response
print('Testing Model')
shared_data_train_run1 = srm_train_run1.transform(run2_list)
shared_data_train_run2 = srm_train_run2.transform(run1_list)

avg_response_train_run1 = sum(shared_data_train_run1)/len(shared_data_train_run1)
avg_response_train_run2 = sum(shared_data_train_run2)/len(shared_data_train_run2)

##################################################################################

# initialize variable that will hold max wva score for each song
max_wvas = np.zeros(16)

for i in range(16):
    print('song number ',str(i))
    # grab start and end time for each song from bound vectors. for SRM data trained on run 1 and tested on run 2, use song name from run 1 to find index for song onset in run 2 bound vector 
    start_run1 = song_bounds_run2[songs_run2.index(songs_run1[i])]
    end_run1   = song_bounds_run2[songs_run2.index(songs_run1[i])+1]
    start_run2 = song_bounds_run1[i]
    end_run2   = song_bounds_run1[i+1]
    # chop song from bold data
    data1 = avg_response_train_run1[:,start_run1:end_run1]
    data2 = avg_response_train_run2[:,start_run2:end_run2]
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
            ROI_WvA[i,j] = within_across

    # store max wva for each song
    max_wvas[i] = np.max(ROI_WvA[i,:])

# compute average max wva across songs
mean_max_wva = np.mean(max_wvas)

# computing average event lengths using song durations divided by number of events
durs_run1_new = durs_run1[:,np.newaxis]

event_lengths = durs_run1_new/K_set

unique_event_lengths = np.unique(event_lengths)
x = event_lengths.ravel()

test_x = np.linspace(min(x), max(x), num=100)

y = ROI_WvA.ravel()
KR = KernelReg(y,x,var_type='c')
KR_w_bw = KernelReg(y,x,var_type='c', bw=KR.bw)
smooth_wva = KR_w_bw.fit(unique_event_lengths)[0]
max_wva = np.max(smooth_wva)

# compute roi's preferred event length in seconds
ROI_pref_sec = unique_event_lengths[np.argmax(smooth_wva)]

inputs = [ROI_WvA,smooth_wva,max_wva,mean_max_wva,ROI_pref_sec]
dct = {}

for i,j in zip(dict_names,inputs):
    dct.setdefault(i,[]).append(j)

np.save(savedir + 'parcel' + roiNum + '_wva_data',dct)
