import nibabel as nib
from moviepy.editor import *
import numpy as np
from scipy.stats import zscore
import h5py
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

data_dir = '/jukebox/norman/jamalw/MES/'
audio_file = data_dir + 'data/songs/My_Favorite_Things.wav'
subj_format = data_dir + 'data/single_song_niftis/My_Favorite_Things/subj%d.nii.gz'
nSubj = 25  # Subjects range from 1 to nSubj
output_suffix = 'My_Favorite_Things'

# Hemodynamic Response Function (from AFNI)
dt = np.arange(0, 15)
p = 8.6
q = 0.547
hrf = np.power(dt / (p * q), p) * np.exp(p - dt / q)

# Don't use the very beginning of the stimulus to compute correlations, to
# ensure that the story has actually started being spoken
ignore_TRs = 2

# Cross-correlation window size (in TRs)
cc_win = 10

# Get indices for whole brain and A1
brain_inds = np.nonzero(nib.load(data_dir + 'data/MNI152_T1_2mm_brain.nii').get_data() > 0)
a1_mask = nib.load(data_dir + 'data/MNI152_T1_2mm_brain.nii').get_data() > 0

# Get metadata about fMRI data
nii_template = nib.load(subj_format % 1)
TR = nii_template.header['pixdim'][4]
nii_shape = nii_template.header.get_data_shape()
nTR = nii_shape[3]

# Load audio, and calculate audio envelope regressor
print('Loading audio...')
clip = AudioFileClip(audio_file)
Fs = clip.fps
samples = clip.to_soundarray()
samples = np.mean(samples, axis=1)
T = np.floor(samples.shape[0] / Fs).astype(int)
rms = np.zeros(T)
for t in range(T):
    rms[t] = np.sqrt(np.mean(np.power(samples[(Fs * t):(Fs * (t + 1))], 2)))
rms_conv = np.convolve(rms, hrf)[:T]
rms_TRs = zscore(np.interp(np.linspace(0, (nTR - 1) * TR, nTR),
                           np.arange(0, T), rms_conv)[ignore_TRs:], ddof=1)

# Compute correlation between rms_TRs and each voxel timecourse
print('Calculating correlations...')
audio_corr = np.zeros(nii_shape[0:3] + (nSubj,))
a1 = np.zeros((nTR, nSubj))
group_mean = np.zeros(nii_shape)
for s in range(nSubj):
    print('   Subj ' + str(s + 1))
    D = nib.load(subj_format % (s + 1)).get_data()
    a1[:, s] = zscore(np.mean(D[a1_mask], axis=0), ddof=1)
    group_mean = group_mean + D / nSubj
    audio_corr[brain_inds + (s,)] = \
        np.matmul(rms_TRs,
                  zscore(D[brain_inds][:, ignore_TRs:], axis=1, ddof=1).T) \
        / (len(rms_TRs) - 1)
audio_corr[np.isnan(audio_corr)] = 0

# Compute correlations for group average timecourses
group_corr = np.zeros(nii_shape[0:3])
group_corr[brain_inds] = \
    np.matmul(rms_TRs,
              zscore(group_mean[brain_inds][:, ignore_TRs:],
                     axis=1, ddof=1).T) \
    / (len(rms_TRs) - 1)

nib.save(nib.Nifti1Image(audio_corr, affine=nii_template.affine),
         data_dir + 'data/AudioCorr_Output/audio_corr_' + output_suffix + '.nii.gz')

nib.save(nib.Nifti1Image(group_corr, affine=nii_template.affine),
         data_dir + 'data/AudioCorr_Output/audio_corr_group_' + output_suffix + '.nii.gz')

f, ax = plt.subplots(2, 2)
ax[0, 0].imshow(group_corr[:, :, 38], vmin=-0.3, vmax=0.3)
ax[0, 0].set_axis_off()
ax[0, 0].set_title('Correlation with audio (axial)')
ax[0, 1].imshow(np.flipud(group_corr[:, 41, :].T), vmin=-0.3, vmax=0.3)
ax[0, 1].set_axis_off()
ax[0, 1].set_title('Correlation with audio (coronal)')

# Compute A1 ISC
ISC = np.zeros(nSubj)
for loo_subj in range(nSubj):
    group = np.mean(a1[:, np.arange(nSubj) != loo_subj], axis=1)
    subj = a1[:, loo_subj]
    ISC[loo_subj] = pearsonr(group, subj)[0]

ISC = np.mean(ISC)
a1_group = np.mean(a1, axis=1)
ax[1, 0].plot(a1)
ax[1, 0].plot(a1_group, 'k', linewidth=2)
ax[1, 0].set_xlabel('Timepoints')
ax[1, 0].set_ylabel('A1 activity')
ax[1, 0].set_title('A1 ISC = ' + str(ISC))

# Compute A1 cross-correlation with audio
cc = np.zeros(2 * cc_win + 1)
a1_group_mid = a1_group[(ignore_TRs + cc_win):(-cc_win)]
for i in range(2 * cc_win + 1):
    if i == 2 * cc_win:
        audio_shift = rms_TRs[i:]
    else:
        audio_shift = rms_TRs[i:(-2 * cc_win + i)]
    cc[i] = pearsonr(audio_shift, a1_group_mid)[0]
ax[1, 1].plot(np.arange(-1 * cc_win, cc_win + 1), cc)
ax[1, 1].set_xlabel('fMRI vs. audio timepoint shift')
ax[1, 1].set_ylabel('fMRI/audio Correlation')

plt.show()

#with h5py.File('a1_' + output_suffix + '.h5', 'w') as hf:
#    hf.create_dataset("subj_data", data=a1)
#    hf.create_dataset("ISC", data=ISC)
#    hf.create_dataset("cc", data=cc)
