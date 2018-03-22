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

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/chris_dartmouth/data/'
song_bounds = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])
songs = ['St Pauls Suite', 'I Love Music', 'Moonlight Sonata', 'Change of the Guard','Waltz of Flowers','The Bird', 'Island', 'Allegro Moderato', 'Finlandia', 'Early Summer', 'Capriccio Espagnole', 'Symphony Fantastique', 'Boogie Stop Shuffle', 'My Favorite Things', 'Blue Monk','All Blues']
songdir = '/jukebox/norman/jamalw/MES/'
all_songs_fn = [songdir + 'data/songs/All_Blues.wav']
idx = 15
song = songs[idx]

spects = []
audio_array_holder = []

# load data
FFMPEG_BIN = "ffmpeg"

def update_line(num, line):
    i = X_VALS[num]
    line[0].set_data( [i, i], [Y_MIN, Y_MAX])
    line[1].set_data( [i, i], [Y_MIN, Y_MAX])
    line[2].set_data( [i, i], [Y_MIN, Y_MAX])
    line[3].set_data( [i, i], [Y_MIN, Y_MAX])
    line[4].set_data( [i, i], [Y_MIN, Y_MAX])
    line[5].set_data( [i, i], [Y_MIN, Y_MAX])
    line[6].set_data( [i, i], [Y_MIN, Y_MAX])
    line[7].set_data( [i, i], [Y_MIN, Y_MAX])

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
#lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
#mfcc *= lift
mfcc_corr = np.corrcoef(mfcc,mfcc)[mfcc.shape[0]:,:mfcc.shape[0]]

# compute normalized mfcc similarity matrix
mfcc_norm = mfcc
mfcc_norm -= (np.mean(mfcc_norm,axis=0) + 1e-8)
mfcc_norm_corr = np.corrcoef(mfcc_norm,mfcc_norm)[mfcc_norm.shape[0]:,:mfcc_norm.shape[0]]

comp_spect = (spect_corr + mfcc_corr + mfcc_norm_corr)/3

# Load in data
train_A1 = np.nan_to_num(zscore(np.load(datadir + 'A1_run1_n25.npy'),axis=1,ddof=1))
test_A1 = np.nan_to_num(zscore(np.load(datadir + 'A1_run2_n25.npy'),axis=1,ddof=1))
train_vmPFC = np.nan_to_num(zscore(np.load(datadir + 'vmPFC_point1_run1_n25.npy'),axis=1,ddof=1))
test_vmPFC = np.nan_to_num(zscore(np.load(datadir + 'vmPFC_point1_run2_n25.npy'),axis=1,ddof=1))

# Convert data into lists where each element is voxels by samples
train_list_A1 = []
test_list_A1 = []
train_list_vmPFC = []
test_list_vmPFC = []

for i in range(0,train_A1.shape[2]):
    train_list_A1.append(train_A1[:,:,i])
    test_list_A1.append(test_A1[:,:,i])

for i in range(0,train_vmPFC.shape[2]):
    train_list_vmPFC.append(train_vmPFC[:,:,i])
    test_list_vmPFC.append(test_vmPFC[:,:,i])

    
# Initialize models
print('Building Model')
srm_A1 = SRM(n_iter=10, features=50)
srm_vmPFC = SRM(n_iter=10, features=10)

# Fit model to training data (run 1)
print('Training Model')
srm_A1.fit(train_list_A1)
srm_vmPFC.fit(train_list_vmPFC)

# Test model on testing data to produce shared response
print('Testing Model')
shared_data_A1 = srm_A1.transform(test_list_A1)
shared_data_vmPFC = srm_vmPFC.transform(test_list_vmPFC)

avg_response_A1 = sum(shared_data_A1)/len(shared_data_A1)
avg_response_vmPFC = sum(shared_data_vmPFC)/len(shared_data_vmPFC)

A1_no_srm = np.mean(test_A1,axis=2)
vmPFC_no_srm = np.mean(test_vmPFC,axis=2)

X_MIN = 0
X_MAX = spect_corr.shape[0] 
Y_MIN = spect_corr.shape[0]
Y_MAX = 0
X_VALS = range(X_MIN, X_MAX);

fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4)
fs = 7

# Plot figures
im1 = ax1.imshow(np.corrcoef(A1_no_srm[:,song_bounds[idx]+3:song_bounds[idx+1]+3].T))
#fig.colorbar(im1,ax=ax1)
ax1.set_title(songs[idx] + ' A1',fontsize=fs)
#ax1.set_xlabel('trs',fontsize=fs,fontweight='bold')
#ax1.set_ylabel('trs',fontsize=fs,fontweight='bold')    
ax1.set_aspect('equal',adjustable='box')
l1 , v1 = ax1.plot(X_MIN, Y_MAX, X_MIN, Y_MIN, linewidth=2, color= 'red')
ax1.set_xlim(X_MIN, X_MAX-2)
ax1.set_ylim(Y_MIN-2, Y_MAX)
ax1.tick_params(axis='both', which='major', labelsize=6)


im2 = ax2.imshow(np.corrcoef(avg_response_A1[:,song_bounds[idx]:song_bounds[idx+1]].T))
#fig.colorbar(im2,ax=ax2)
ax2.set_title(songs[idx] + ' A1 srm',fontsize=fs)
#ax2.set_xlabel('trs',fontsize=fs,fontweight='bold')
#ax2.set_ylabel('trs',fontsize=fs,fontweight='bold')    
ax2.set_aspect('equal',adjustable='box')
l2 , v2 = ax2.plot(X_MIN, Y_MAX, X_MIN, Y_MIN, linewidth=2, color= 'red')
ax2.set_xlim(X_MIN, X_MAX-2)
ax2.set_ylim(Y_MIN-2, Y_MAX)
ax2.tick_params(axis='both', which='major', labelsize=6)

im3 = ax3.imshow(np.corrcoef(vmPFC_no_srm[:,song_bounds[idx]:song_bounds[idx+1]].T))
#fig.colorbar(im3,ax=ax3)
ax3.set_title(songs[idx] + ' vmPFC',fontsize=fs)
#ax3.set_xlabel('trs',fontsize=fs,fontweight='bold')
#ax3.set_ylabel('trs',fontsize=fs,fontweight='bold')    
ax3.set_aspect('equal',adjustable='box')
l3 , v3 = ax3.plot(X_MIN, Y_MAX, X_MAX, Y_MIN, linewidth=2, color= 'red')
ax3.set_xlim(X_MIN, X_MAX-2)
ax3.set_ylim(Y_MIN-2, Y_MAX)
ax3.tick_params(axis='both', which='major', labelsize=6)

im4 = ax4.imshow(np.corrcoef(avg_response_vmPFC[:,song_bounds[idx]+5:song_bounds[idx+1]+5].T))
#fig.colorbar(im4,ax=ax4)
ax4.set_title(songs[idx] + ' vmPFC srm',fontsize=fs)
#ax4.set_xlabel('trs',fontsize=fs,fontweight='bold')
#ax4.set_ylabel('trs',fontsize=fs,fontweight='bold')    
ax4.set_aspect('equal',adjustable='box')
l4 , v4 = ax4.plot(X_MIN, Y_MAX, X_MAX, Y_MIN, linewidth=2, color= 'red')
ax4.set_xlim(X_MIN, X_MAX-2)
ax4.set_ylim(Y_MIN-2, Y_MAX)
ax4.tick_params(axis='both', which='major', labelsize=6)

im5 = ax5.imshow(spect_corr)
#fig.colorbar(im5,ax=ax5)
ax5.set_title('Spectral Similarity',fontsize=fs)
ax5.set_aspect('equal',adjustable='box')
l5 , v5 = ax5.plot(X_MIN, Y_MAX, X_MAX, Y_MIN, linewidth=2, color= 'red')
ax5.set_xlim(X_MIN, X_MAX-2)
ax5.set_ylim(Y_MIN-2, Y_MAX)
ax5.tick_params(axis='both', which='major', labelsize=6)

im6 = ax6.imshow(mfcc_corr)
#fig.colorbar(im6,ax=ax6)
ax6.set_title('MFCC Similarity',fontsize=fs)
ax6.set_aspect('equal',adjustable='box')
l6 , v6 = ax6.plot(X_MIN, Y_MAX, X_MAX, Y_MIN, linewidth=2, color= 'red')
ax6.set_xlim(X_MIN, X_MAX-2)
ax6.set_ylim(Y_MIN-2, Y_MAX)
ax6.tick_params(axis='both', which='major', labelsize=6)

im7 = ax7.imshow(mfcc_norm_corr)
#fig.colorbar(im7,ax=ax7)
ax7.set_title('Norm-MFCC Similarity',fontsize=fs)
ax7.set_aspect('equal',adjustable='box')
l7 , v7 = ax7.plot(X_MIN, Y_MAX, X_MAX, Y_MIN, linewidth=2, color= 'red')
ax7.set_xlim(X_MIN, X_MAX-2)
ax7.set_ylim(Y_MIN-2, Y_MAX)
ax7.tick_params(axis='both', which='major', labelsize=6)

im8 = ax8.imshow(comp_spect)
#fig.colorbar(im8,ax=ax8)
ax8.set_title('Composite Similarity',fontsize=fs)
ax8.set_aspect('equal',adjustable='box')
l8 , v8 = ax8.plot(X_MIN, Y_MAX, X_MAX, Y_MIN, linewidth=2, color= 'red')
ax8.set_xlim(X_MIN, X_MAX-2)
ax8.set_ylim(Y_MIN-2, Y_MAX)
ax8.tick_params(axis='both', which='major', labelsize=6)

l = [l1,l2,l3,l4,l5,l6,l7,l8]

line_anim = animation.FuncAnimation(fig, update_line, len(X_VALS),   
                                    fargs=(l, ), interval=100,
                                    blit=True, repeat=False)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
line_anim.save('test_video.mp4', writer=writer)
print('video saved')


