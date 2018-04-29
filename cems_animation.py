import numpy as np
from scipy.signal import spectrogram, gaussian, convolve
import matplotlib.pyplot as plt
import glob
from scipy.stats import norm, zscore, pearsonr
from pydub import AudioSegment
from scipy.fftpack import dct
import matplotlib.animation as animation
import brainiak.eventseg.event
from sklearn import decomposition
from brainiak.funcalign.srm import SRM
import nibabel as nib
from scipy.io import wavfile
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

song = int(sys.argv[1]) -1
datadir = '/tigress/jamalw/MES/prototype/link/scripts/chris_dartmouth/data/'
ann_dirs = '/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'

song_bounds = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

durs = np.array([90,180,180,90,135,180,180,225,225,135,90,135,225,225,90,135])

nTR = durs[song]

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

songdir = '/tigress/jamalw/MES/'

human_bounds = np.load(ann_dirs + songs[song] + '/' + songs[song] + '_beh_seg.npy')

human_bounds = np.append(0,np.append(human_bounds,durs[song]))

all_songs_fn = [songdir + 'data/songs/Change_of_the_Guard.wav']

hrf = 5

start = song_bounds[song] + hrf
end   = song_bounds[song + 1] + hrf

spects = []
audio_array_holder = []

# load data
FFMPEG_BIN = "ffmpeg"

def update_line(num, line):
    i = X_VALS[num]
    line[0].set_data( [i, i], [Y_MIN, Y_MAX])
    #line[1].set_data( [i, i], [Y_MIN, Y_MAX])
    #line[2].set_data( [i, i], [Y_MIN, Y_MAX])

    return line 


for j in range(len(all_songs_fn)):
    rate, audio = wavfile.read(all_songs_fn[j])
    audio_array = np.mean(audio,axis=1)
    print('computing spectrogram')     
    f,t,spect = spectrogram(audio_array,44100)
    spects.append(spect)
    print('spectrogram computed')

w = np.round(spect.shape[1]/(len(audio_array)/44100))
output = np.zeros((spect.shape[0],int(np.round((t.shape[0]/w)))))
forward_idx = np.arange(0,len(t) + 1,w)
num_ceps = 12

for i in range(len(forward_idx)):
    if spect[:,int(forward_idx[i]):int(forward_idx[i])+int(w)].shape[1] != w:
        continue  
    else: 
        output[:,i] = np.mean(spect[:,int(forward_idx[i]):int(forward_idx[i])+int(w)],axis=1).T

# compute similarity matrix for spectrogram
spect_corr = np.corrcoef(output.T,output.T)[output.shape[1]:,:output.shape[1]]

# compute similarity matrix for mfcc
mfcc = dct(output.T, type=2,axis=1,norm='ortho')[:,1:(num_ceps + 1)]
(nframes, ncoeff) = mfcc.shape
n = np.arange(ncoeff)
cep_lifter = 12
lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
mfcc *= lift
mfcc_corr = np.corrcoef(mfcc,mfcc)[mfcc.shape[0]:,:mfcc.shape[0]]

# compute normalized mfcc similarity matrix
mfcc_norm = mfcc
mfcc_norm -= (np.mean(mfcc_norm,axis=0) + 1e-8)
mfcc_norm_corr = np.corrcoef(mfcc_norm,mfcc_norm)[mfcc_norm.shape[0]:,:mfcc_norm.shape[0]]

comp_spect = (spect_corr + mfcc_corr + mfcc_norm_corr)/3

# Load in data
train_roi_1 = np.nan_to_num(zscore(np.load(datadir + 'A1_run1_n25.npy'),axis=1,ddof=1))
test_roi_1 = np.nan_to_num(zscore(np.load(datadir + 'A1_run2_n25.npy'),axis=1,ddof=1))
train_roi_2 = np.nan_to_num(zscore(np.load(datadir + 'zstats_human_bounds_superior_parietal_tight_run1_n25.npy'),axis=1,ddof=1))
test_roi_2 = np.nan_to_num(zscore(np.load(datadir + 'zstats_human_bounds_superior_parietal_tight_run2_n25.npy'),axis=1,ddof=1))

# Convert data into lists where each element is voxels by samples
train_list_roi_1 = []
test_list_roi_1 = []
train_list_roi_2 = []
test_list_roi_2 = []

for i in range(0,train_roi_1.shape[2]):
    train_list_roi_1.append(train_roi_1[:,:,i])
    test_list_roi_1.append(test_roi_1[:,:,i])

for i in range(0,train_roi_2.shape[2]):
    train_list_roi_2.append(train_roi_2[:,:,i])
    test_list_roi_2.append(test_roi_2[:,:,i])

    
# Initialize models
print('Building Model')
srm_roi_1 = SRM(n_iter=50, features=5)
srm_roi_2 = SRM(n_iter=50, features=5)

# Fit model to training data (run 1)
print('Training Model')
srm_roi_1.fit(train_list_roi_1)
srm_roi_2.fit(train_list_roi_2)

# Test model on testing data to produce shared response
print('Testing Model')
shared_data_roi_1 = srm_roi_1.transform(test_list_roi_1)
shared_data_roi_2 = srm_roi_2.transform(test_list_roi_2)

avg_response_roi_1 = sum(shared_data_roi_1)/len(shared_data_roi_1)
avg_response_roi_2 = sum(shared_data_roi_2)/len(shared_data_roi_2)

# Fit to ROI 1
ev1 = brainiak.eventseg.event.EventSegment(len(human_bounds) - 1)
ev1.fit(avg_response_roi_1[:,start:end].T)

bounds1 = np.where(np.diff(np.argmax(ev1.segments_[0], axis=1)))[0]

# Fit to ROI 2
ev2 = brainiak.eventseg.event.EventSegment(len(human_bounds) - 1)
ev2.fit(avg_response_roi_2[:,start:end].T)

bounds2 = np.where(np.diff(np.argmax(ev2.segments_[0], axis=1)))[0]

# Fit to song data
ev3 = brainiak.eventseg.event.EventSegment(len(human_bounds) - 1)
ev3.fit(mfcc)

bounds3 = np.where(np.diff(np.argmax(ev3.segments_[0], axis=1)))[0]

X_MIN = 0
X_MAX = spect_corr.shape[0] 
Y_MIN = spect_corr.shape[0]
Y_MAX = 0
X_VALS = range(X_MIN, X_MAX);

#fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
fs = 18

## Plot figures
#im1 = ax1.imshow(np.corrcoef(avg_response_roi_1[:,start:end].T))
##fig.colorbar(im1,ax=ax1)
#ax1.set_title(songs[song] + ' A1',fontsize=fs)
#ax1.set_xlabel('TRs',fontsize=fs,fontweight='bold')
#ax1.set_ylabel('TRs',fontsize=fs,fontweight='bold')    
#ax1.set_aspect('equal',adjustable='box')
#l1 , v1 = ax1.plot(X_MIN, Y_MAX, X_MIN, Y_MIN, linewidth=2, color= 'red')
#ax1.set_xlim(X_MIN, X_MAX-2)
#ax1.set_ylim(Y_MIN-2, Y_MAX)
#ax1.tick_params(axis='both', which='major', labelsize=6)
#
## add annotations
#bounds_aug = np.concatenate(([0],bounds1,[nTR]))
#for i in range(len(bounds_aug)-1):
#    rect1 = patches.Rectangle((bounds_aug[i],bounds_aug[i]),bounds_aug[i+1]-bounds_aug[i],bounds_aug[i+1]-bounds_aug[i],linewidth=3,edgecolor='w',facecolor='none',label='Model Fit')
#    ax1.add_patch(rect1)
#
#for i in range(len(human_bounds)-1):
#    rect2 = patches.Rectangle((human_bounds[i],human_bounds[i]),human_bounds[i+1]-human_bounds[i],human_bounds[i+1]-human_bounds[i],linewidth=3,edgecolor='k',facecolor='none',label='Human Annotations')
#    ax1.add_patch(rect2)
#
fig = plt.figure()
plt.imshow(np.corrcoef(avg_response_roi_2[:,start:end].T))
ax2 = plt.gca()
#fig.colorbar(im2,ax=ax2)
ax2.set_title('Change of the Guard', fontsize=fs)
ax2.set_xlabel('TRs',fontsize=fs,fontweight='bold')
ax2.set_ylabel('TRs',fontsize=fs,fontweight='bold')    
ax2.set_aspect('equal',adjustable='box')
l2 , v2 = ax2.plot(X_MIN, Y_MAX, X_MIN, Y_MIN, linewidth=2, color= 'red')
ax2.set_xlim(X_MIN, X_MAX-2)
ax2.set_ylim(Y_MIN-2, Y_MAX)
ax2.tick_params(axis='both', which='major', labelsize=15)

# add annotations
bounds_aug = np.concatenate(([0],bounds2,[nTR]))
for i in range(len(bounds_aug)-1):
    rect1 = patches.Rectangle((bounds_aug[i],bounds_aug[i]),bounds_aug[i+1]-bounds_aug[i],bounds_aug[i+1]-bounds_aug[i],linewidth=5,edgecolor='w',facecolor='none',label='Model Fit')
    ax2.add_patch(rect1)

for i in range(len(human_bounds)-1):
    rect2 = patches.Rectangle((human_bounds[i],human_bounds[i]),human_bounds[i+1]-human_bounds[i],human_bounds[i+1]-human_bounds[i],linewidth=5,edgecolor='k',facecolor='none',label='Human Annotations')
    ax2.add_patch(rect2)

plt.legend(handles=[rect1,rect2])

#im3 = ax3.imshow(mfcc.T)
##fig.colorbar(im3,ax=ax3)
#ax3.set_title(songs[song] + ' MFCC',fontsize=fs)
#ax3.set_xlabel('TRs',fontsize=fs,fontweight='bold')
#ax3.set_ylabel('TRs',fontsize=fs,fontweight='bold')    
#ax3.set_aspect('equal',adjustable='box')
#l3 , v3 = ax3.plot(0, mfcc.shape[1], 0, 0, linewidth=2, color= 'red')
#ax3.set_xlim(0, mfcc.shape[0] - 2)
#ax3.set_ylim(mfcc.shape[1] - 2, 0)
#ax3.tick_params(axis='both', which='major', labelsize=6)
#
# add annotations
#bounds_aug = np.concatenate(([0],bounds3,[nTR]))
#for i in range(len(bounds_aug)-1):
#    rect1 = patches.Rectangle((bounds_aug[i],bounds_aug[i]),bounds_aug[i+1]-bounds_aug[i],bounds_aug[i+1]-bounds_aug[i],linewidth=3,edgecolor='w',facecolor='none',label='Model Fit')
#    ax3.add_patch(rect1)

#for i in range(len(human_bounds)-1):
#    rect2 = patches.Rectangle((human_bounds[i],human_bounds[i]),human_bounds[i+1]-human_bounds[i],human_bounds[i+1]-human_bounds[i],linewidth=3,edgecolor='k',facecolor='none',label='Human Annotations')
#    ax3.add_patch(rect2)

#for i in human_bounds[1:-1]:
#    plt.axvline(x=i,color='k',linewidth=3)

plt.tight_layout()

l = [l2]

line_anim = animation.FuncAnimation(fig, update_line, len(X_VALS),   
                                    fargs=(l, ), interval=100,
                                    blit=True, repeat=False)


FFwriter = animation.FFMpegWriter(fps=1,extra_args=['-vcodec','libx264'])
#line_anim.save('basic_animation.mp4', writer = FFwriter)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
line_anim.save('cems_animation.mp4', writer=writer)
print('video saved')


