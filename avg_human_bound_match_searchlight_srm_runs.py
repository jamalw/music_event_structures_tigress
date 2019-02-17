import numpy as np
import glob
import nibabel as nib


nii_template = nib.load('/tigress/jamalw/MES/data/trans_filtered_func_data.nii')

datadir = '/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_wva/' 
#fn_raw_run1 = glob.glob(datadir + '*raw_wva_bound_match_across_songs_run1.nii.gz')
#fn_raw_run2 = glob.glob(datadir + '*raw_wva_bound_match_across_songs_run2.nii.gz')
fn_z_run1   = glob.glob(datadir + '*z_wva_bound_match_across_songs_run1_no_motion.nii.gz')
fn_z_run2   = glob.glob(datadir + '*z_wva_bound_match_across_songs_run2_no_motion.nii.gz')
    
#raw_data_run1 = nib.load(fn_raw_run1[0]).get_data()
#raw_data_run2 = nib.load(fn_raw_run2[0]).get_data()
z_data_run1 = nib.load(fn_z_run1[0]).get_data()
z_data_run2 = nib.load(fn_z_run2[0]).get_data()
    
#avg_both_raw_runs = (raw_data_run1 + raw_data_run2)/2 
avg_both_z_runs = (z_data_run1 + z_data_run2)/2   

# Save all raw averages
#maxval = np.max(avg_both_raw_runs)
#minval = np.min(avg_both_raw_runs)
#img = nib.Nifti1Image(avg_both_raw_runs, affine=nii_template.affine)
#img.header['cal_min'] = minval
#img.header['cal_max'] = maxval
#nib.save(img,datadir + 'avg_both_raw_runs_srmk_30.nii.gz')  

# Save all zscore averages
maxval = np.max(avg_both_z_runs)
minval = np.min(avg_both_z_runs)
img = nib.Nifti1Image(avg_both_z_runs, affine=nii_template.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + 'avg_both_z_runs_srmk_30.nii.gz')  
    

 
