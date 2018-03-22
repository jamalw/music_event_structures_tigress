import numpy as np
from scipy import special
import nibabel as nib
import sys
import scipy.stats as st

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/'
filename = sys.argv[1]

#z_scores = np.load(filename)
#z_scores_reshaped = np.reshape(z_scores,(91*109*91))
z_scores = nib.load(datadir + filename).get_data()
z_scores_reshaped = np.nan_to_num(np.reshape(z_scores,(91*109*91)))
z_scores_reshaped2 = np.nan_to_num(np.reshape(z_scores,(91*109*91)))

mask = z_scores_reshaped != 0

z_scores_reshaped[mask] = st.norm.sf(z_scores_reshaped[mask])
p_values = z_scores_reshaped
z_scores_reshaped2[mask] = -np.log(p_values[mask])
neg_log_p_values = z_scores_reshaped2
p_values = np.reshape(p_values,(91,109,91))
neg_log_p_values = np.reshape(neg_log_p_values,(91,109,91))

min1 = np.min(p_values[~np.isnan(p_values)])
max1 = np.max(p_values[~np.isnan(p_values)])
img1 = nib.Nifti1Image(p_values, np.eye(4))
img1.header['cal_min'] = min1
img1.header['cal_max'] = max1
nib.save(img1,datadir + 'HMM_searchlight_K16_w5_n25_pvals.nii.gz')

min2 = np.min(neg_log_p_values[~np.isnan(neg_log_p_values)])
max2 = np.max(neg_log_p_values[~np.isnan(neg_log_p_values)])
img2 = nib.Nifti1Image(neg_log_p_values, np.eye(4))
img2.header['cal_min'] = min2
img2.header['cal_max'] = max2
nib.save(img2,datadir + 'HMM_searchlight_K16_w5_n25_neg_log_pvals.nii.gz')

#np.save('audio_env_by_nii_results_n25_pvals',p_values)
