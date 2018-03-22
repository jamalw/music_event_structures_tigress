import numpy as np
import sys
import nibabel as nib

subj = ['MES_022817_0','MES_030217_0','MES_032117_1','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

datadir = '/jukebox/norman/jamalw/MES/'

avg_run1 = np.zeros((91,109,91,2511))
avg_run2 = np.zeros((91,109,91,2511))

for i in range(len(subj)):
     run1 = nib.load(datadir + '/subjects/' + subj[i] + '/analysis/run1.feat/trans_filtered_func_data.nii')
     run1 = np.array(run1.get_data()[:,:,:,0:2511]) 
     avg_run1 += run1/len(subj)

     run2 = nib.load(datadir + '/subjects/' + subj[i] + '/analysis/run2.feat/trans_filtered_func_data.nii')
     run2 = np.array(run2.get_data()[:,:,:,0:2511])
     avg_run2 += run2/len(subj)

min1 = np.min(avg_run1[~np.isnan(avg_run1)])
max1 = np.max(avg_run1[~np.isnan(avg_run1)])
img1 = nib.Nifti1Image(avg_run1, np.eye(4))
img1.header['cal_min'] = min1
img1.header['cal_max'] = max1
nib.save(img1,'trans_filtered_func_avg_run1.nii.gz')

min2 = np.min(avg_run2[~np.isnan(avg_run2)])
max2 = np.max(avg_run2[~np.isnan(avg_run2)])
img2 = nib.Nifti1Image(avg_run2, np.eye(4))
img2.header['cal_min'] = min2
img2.header['cal_max'] = max2
nib.save(img2,'trans_filtered_func_avg_run2.nii.gz')


