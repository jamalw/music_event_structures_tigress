import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
from nilearn.input_data import NiftiMasker
import soundfile as sf
import pickle

subj = sys.argv[1]
roi  = sys.argv[2]
roifilename = sys.argv[3]

datadir = '/Users/jamalw/Desktop/PNI/music_event_structures/'

# set data filenames
mask_filename  = datadir + roifilename
fmri1_filename = datadir + 'subjects/' + subj + '/trans_filtered_func_data1.nii'
fmri2_filename = datadir + 'subjects/' + subj + '/trans_filtered_func_data2.nii'

# mask data
masker = NiftiMasker(mask_img=mask_filename, standardize=True)
fmri1_masked = masker.fit_transform(fmri1_filename)
fmri1_masked = stats.zscore(fmri1_masked.T)
fmri2_masked = masker.fit_transform(fmri2_filename)
fmri2_masked = stats.zscore(fmri2_masked.T)

# load song data
songs1 = np.load(datadir + 'all_songs_part1.npy')
songs2 = np.load(datadir + 'all_songs_part2.npy')

# get song durations
songs1Dur = []
songs2Dur = []

for i in range(len(songs1)):
    f1 = sf.SoundFile(songs1[i])
    f2 = sf.SoundFile(songs2[i])
    songs1Dur = np.append(songs1Dur,np.round(len(f1)/f1.samplerate))
    songs2Dur = np.append(songs2Dur,np.round(len(f2)/f2.samplerate))

# slice functional scan according to song model
func_sliced1 = []
func_sliced2 = []
data1 = fmri1_masked
data2 = fmri2_masked
for i in range(len(songs1Dur)):
    func_sliced1.append([])
    func_sliced2.append([])
    func_sliced1[i].append(data1[:,0:songs1Dur[i]])
    func_sliced2[i].append(data2[:,0:songs2Dur[i]])
    data1 = data1[:,songs1Dur[i]:]
    data2 = data2[:,songs2Dur[i]:]

# create subject specific song model for both experiments
exp1 = np.array([7, 12, 15,  2,  1,  0,  9,  3,  4,  5,  6, 13, 10,  8, 11, 14])
exp2 = np.array([3, 15,  4, 13,  2, 11,  0,  6,  7, 14,  1,  5, 10,  8, 12, 9])

# reorder func data according to genre model
reorder1 = []
reorder2 = []
for i in range(len(exp1)):
    reorder1.append([])
    reorder2.append([])

for i in range(len(exp1)):
    reorder1[exp1[i]] = func_sliced1[i][0]
    reorder2[exp2[i]] = func_sliced2[i][0]

pickle.dump(reorder1, open(datadir + 'subjects/' + subj + '/' + 'reorder1' + roi + '.p','wb'))
pickle.dump(reorder2, open(datadir + 'subjects/' + subj + '/' + 'reorder2' + roi + '.p','wb'))
