import numpy as np
from scipy import special
import scipy.stats as st

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_wva/'

filename = 'avg_perms_both_runs_across_songs.npy'

# load data
z_scores = np.load(datadir + filename)

# pre-allocate array
pvals = np.empty((91,109,91,1001))

for i in range(data.shape[3]):
    # reshape zscores for real zscores and permutations separately
    z_scores_reshaped = np.reshape(z_scores[i],(91*109*91))
    # mask array by grabbing on non-zero values
    mask = z_scores_reshaped != 0
    # convert zscores to p-values
    z_scores_reshaped[mask] = st.norm.sf(z_scores_reshaped[mask])
    # reshape zscores and store in respective array position  
    pvals[i] = np.reshape(z_scores_reshaped,(91,109,91))

np.save(data + 'avg_perms_both_runs_across_songs_pvals', p_vals)
