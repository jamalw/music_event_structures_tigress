from scipy.stats import norm,zscore,pearsonr,stats
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# Custom mean estimator with Fisher z transformation for correlations
def fisher_mean(correlation, axis=None):
    return np.tanh(np.mean(np.arctanh(correlation), axis=axis))

datadir = "/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/parcels/Schaefer100/"

human_bounds_dir = "/tigress/jamalw/MES/prototype/link/scripts/data/beh/annotations/"

songs = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

durs = np.array([225,89,180,134,90,180,134,90,179,135,224,89,224,225,179,134])

for i in range(len(songs)):
    # initialize timeseries
    hmm_run1_series = np.zeros((durs[i]))
    hmm_run2_series = np.zeros((durs[i]))
    hb_series = np.zeros((durs[i])) 
    # load hmm and human boundaries
    hmm_bounds_run1 = np.around(np.load(datadir + songs[i] + '/hmm_bounds_rA1_run1.npy'))
    hmm_bounds_run2 = np.around(np.load(datadir + songs[i] + '/hmm_bounds_rA1_run2.npy'))
    human_bounds = np.load(human_bounds_dir + songs[i] + '_beh_seg.npy')
    # populate timeseries with boundaries
    for b in range(len(human_bounds)):
        hmm_run1_series[int(hmm_bounds_run1[b])] = 1
        hmm_run2_series[int(hmm_bounds_run2[b])] = 1
        hb_series[int(human_bounds[b])] = 1   

    cross_corr_run1 = signal.correlate(hmm_run1_series,hb_series)
    cross_corr_run2 = signal.correlate(hmm_run2_series,hb_series)

    plt.plot(cross_corr_run1,color='k')
    plt.plot(cross_corr_run2,color='g')


    x=10
    
