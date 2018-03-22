import numpy as np
from scipy import special
import nibabel as nib
import sys
import scipy.stats as st

filename = sys.argv[1]
searchlight_dir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/'

p_values = nib.load(searchlight_dir + filename).get_data()
p_values_reshaped = np.reshape(p_values,(91*109*91))

mask = p_values_reshaped != 0

p_values_reshaped[mask] = -np.log(p_values_reshaped[mask])
neg_log_p_values = np.reshape(p_values_reshaped,(91,109,91))

min2 = np.min(neg_log_p_values[~np.isnan(neg_log_p_values)])
max2 = np.max(neg_log_p_values[~np.isnan(neg_log_p_values)])
img2 = nib.Nifti1Image(neg_log_p_values, np.eye(4))
img2.header['cal_min'] = min2
img2.header['cal_max'] = max2
nib.save(img2, searchlight_dir + filename + '_n25_neg_log_pvals.nii.gz')
