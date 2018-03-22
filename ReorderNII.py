import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
from nilearn.input_data import NiftiMasker
import soundfile as sf
import pickle
from nilearn.image import load_img
import nibabel as nib

subjs = ['MES_022817_0','MES_030217_0', 'MES_032117_1','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0', 'MES_041417_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

#subjs = ['MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

datadir = '/jukebox/norman/jamalw/MES/'

# create subject specific song model for both experiments
exp1 = np.array([7, 12, 15,  2,  1,  0,  9,  3,  4,  5,  6, 13, 10,  8, 11, 14])
exp2 = np.array([3, 15,  4, 13,  2, 11,  0,  6,  7, 14,  1,  5, 10,  8, 12, 9])

for j in range(0,len(subjs)):
    # set data filenames
    fmri1_filename = datadir + 'subjects/' + subjs[j] + '/analysis/run1.feat/trans_filtered_func_data.nii'
    fmri2_filename = datadir + 'subjects/' + subjs[j] + '/analysis/run2.feat/trans_filtered_func_data.nii'

    # get song durations
    songs1Dur = np.load(datadir + 'data/' + 'songs1Dur.npy')
    songs2Dur = np.load(datadir + 'data/' + 'songs2Dur.npy') 

    # slice functional scan according to song model
    data1 = load_img(datadir + 'subjects/' + subjs[j] + '/analysis/run1.feat/trans_filtered_func_data.nii')
    data2 = load_img(datadir + 'subjects/' + subjs[j] + '/analysis/run2.feat/trans_filtered_func_data.nii')
    data1 = data1.get_data()
    data2 = data2.get_data()
    data1_reshape = stats.zscore(np.reshape(data1,(91*109*91,data1.shape[3])),axis=1,ddof=1)
    data2_reshape = stats.zscore(np.reshape(data2,(91*109*91,data2.shape[3])),axis=1,ddof=1)
    data1 = np.reshape(data1_reshape,(91,109,91,data1.shape[3]))
    data2 = np.reshape(data2_reshape,(91,109,91,data2.shape[3]))

    func_sliced1 = []
    func_sliced2 = []
    for i in range(len(songs1Dur)):
        func_sliced1.append([])
        func_sliced2.append([])

    for i in range(0,len(songs1Dur)):
        func_sliced1[i].append(data1[:,:,:,0:songs1Dur[i]])
        func_sliced2[i].append(data2[:,:,:,0:songs2Dur[i]])
        data1 = data1[:,:,:,songs1Dur[i]:]
        data2 = data2[:,:,:,songs2Dur[i]:]

    # reorder func data according to genre model
    reorder1 = []
    reorder2 = []
    for i in range(len(exp1)):
        reorder1.append([])
        reorder2.append([])

    for i in range(len(exp1)):
        reorder1[exp1[i]] = func_sliced1[i][0]
        reorder2[exp2[i]] = func_sliced2[i][0]

    # concatenate matrices to get back to single matrix VxT
    reorder1_full = np.concatenate((reorder1),3)
    reorder2_full = np.concatenate((reorder2),3)

    # average matrices together to get one averaged VxT matrix
    subj_avg_reordered_data = (reorder1_full + reorder2_full)/2 

    np.save(datadir + 'subjects/' + subjs[j] + '/data/' + 'avg_reordered_both_runs',subj_avg_reordered_data)
    min1 = np.min(subj_avg_reordered_data[~np.isnan(subj_avg_reordered_data)])
    max1 = np.max(subj_avg_reordered_data[~np.isnan(subj_avg_reordered_data)])
    img1 = nib.Nifti1Image(subj_avg_reordered_data, np.eye(4))
    img1.header['cal_min'] = min1
    img1.header['cal_max'] = max1
    nib.save(img1, datadir + 'subjects/' + subjs[j] + '/data/' + 'avg_reordered_both_runs.nii.gz')    

#avgsubjData1 = []
#avgsubjData2 = []
#for j in range(len(reorder1)):
#    avgsubjData1.append(np.mean(reorder1[j],3))
#    avgsubjData2.append(np.mean(reorder2[j],3))

#reorder1_img = np.concatenate([aux[...,np.newaxis] for aux in avgsubjData1],axis=3)
#reorder1_img = reorder1_img[...,0]

#reorder2_img = np.concatenate([aux[...,np.newaxis] for aux in avgsubjData2],axis=3)
#reorder2_img = reorder2_img[...,0]

#img1 = nib.Nifti1Image(reorder1_img, np.eye(4))
#img2 = nib.Nifti1Image(reorder2_img, np.eye(4))

#nib.save(img1, datadir + 'subjects/' + subj + '/data/' + 'avg_reorder1.nii.gz')
#nib.save(img2, datadir + 'subjects/' + subj + '/data/' + 'avg_reorder2.nii.gz')
