import pandas as pd
import numpy as np
import csv
import peakutils
import matplotlib.pyplot as plt
import nilearn

#songs = ['Pauls','I Love Music','Moonlight Sonata','Change of the Guard','Waltz of the Flowers','The Bird','Island','Allegro Moderato','Finlandia','Early Summer','Capriccio Espagnole','Symphony - Fantastique','Boogie Stop Shuffle','My Favorite Things','Blue Monk','All Blues']

songs = ['Boogie Stop Shuffle']

subjs = ['Elise','SS_021618_0','SS_021618_1','SS_021718_0','SS_021718_2','SS_021818_0','SS_021818_1','SS_021818_2','SS_030118_0','SS_030218_0']

#durs = np.array([90,180,180,89,134,179,180,224,225,134,90,135,225,225,89,134])

durs = np.array([225])

datadir = '/tigress/jamalw/MES/prototype/link/scripts/data/'
# initialize our final output: which is a list where each element is a matrix corresponding to a song. the shape of each matrix is nSubjs x nTRs and contains ones at TRs where there was a button press and zeros everywhere else
allPressTimes = []
bounds = []

for i in range(len(songs)):
    # initialize song specific button press time holder (nSubjs x nTrs)
    songPressTimes = np.array([])
    song_bounds = []
    for j in range(len(subjs)):
        # load in subject data. here i load the full csv file AND timestamps column only into two separate variables.  
        data = pd.read_csv(datadir + 'beh/' + subjs[j] + '/' + subjs[j] + '_subjective_segmentation.csv',na_values=" NaN")
        timestamps = pd.read_csv(datadir + 'beh/' + subjs[j] + '/' + subjs[j] + '_subjective_segmentation.csv',index_col='timestamp')

        # grab song end and start times (with event type and info)
        song_end_and_start = data[(data['info'].str.contains(songs[i],na=False))]

        # store start and end time separately
        start = song_end_and_start['timestamp'].iloc[0]
        end = song_end_and_start['timestamp'].iloc[1]

        # compute number of TRs and initialize subject-specific button press vector (vector of zeros that will be replaced with ones according to timestep index) 
        #nTRs = np.round(end - start)
        nTRs = durs[i]
        subj_press_vector = np.zeros((int(nTRs)))
        # grab full range of rows and colums between start and end time 
        button_presses_full = timestamps[start:end]

        # grab only button press timesteps including song start and end time
        button_presses = np.round(button_presses_full[(button_presses_full['info'].str.contains('j',na=False))].index.values.tolist())

        # compute distances between button presses and sum to get cumulative press time distribution
        distance_btwn_presses = np.cumsum(np.diff(button_presses)[:-1]).astype(int)
        if distance_btwn_presses[-1] == nTRs:
            distance_btwn_presses[-1] = nTRs - 1
        song_bounds.append([distance_btwn_presses]) 
        # replace zeros with ones where presses occurred and store vector in song specific button press time holder
        subj_press_vector[distance_btwn_presses] = 1
        songPressTimes = np.concatenate([songPressTimes,subj_press_vector],axis=0)
    bounds.append(song_bounds)
    allPressTimes.append(np.reshape(songPressTimes,(len(subjs),int(nTRs))))

all_songs_indexes = []

for i in range(len(durs)):
    combined = np.zeros((durs[i],1))
    for t in np.arange(0,len(combined)):
        combined[t] = sum([min(abs(x[0]-(t+1)))<=3 for x in bounds[i]])
    combined = combined.reshape((durs[i]))
    #subj_sums = np.sum(combined,axis=0)
    plt.figure(i)
    plt.plot(combined)
    plt.xlabel('time(s)',fontsize=18)
    plt.ylabel('Sum of Presses',fontsize=18)
    plt.ylim([0,len(subjs)])
    plt.tight_layout()
    plt.title(songs[i],fontsize=18)    
    indexes = peakutils.indexes(combined,thres=0.5, min_dist=5)
    all_songs_indexes.append([indexes])
    #plt.savefig('/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/plots/subjective_sums_' + songs[i])

for i in range(len(all_songs_indexes)):
    peak_vector = np.zeros((durs[i]))
    peak_vector[all_songs_indexes[i][0]] = 1
#    plt.figure(i)
#    plt.plot(peak_vector)
#    plt.title(songs[i],fontsize=18)
#    plt.xlabel('time(s)',fontsize=18)
#    plt.ylabel('boundary(0 or 1)',fontsize=18)
#    plt.tight_layout()
#    plt.savefig('/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/plots/subjective_peaks_' + songs[i])    

#out = csv.writer(open("beh_peaks.csv","w"), delimiter=',',quoting=csv.QUOTE_ALL)
#out.writerow(all_songs_indexes)

