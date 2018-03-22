#import deepdish as dd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import numpy as np
import brainiak.eventseg.event
from scipy.stats import norm, zscore, pearsonr, stats
from scipy.signal import gaussian, convolve
from sklearn import decomposition
import numpy as np
from brainiak.funcalign.srm import SRM

datadir = '/tigress/jamalw/MES/prototype/link/scripts/chris_dartmouth/data/'

# Load in data
train = np.nan_to_num(stats.zscore(np.load(datadir + 'angular_gyrus_raw_hmm_searchlight_run1_n25.npy'),axis=1,ddof=1))
test = np.nan_to_num(stats.zscore(np.load(datadir + 'angular_gyrus_raw_hmm_searchlight_run2_n25.npy'),axis=1,ddof=1))

# Convert data into lists where each element is voxels by samples
train_list = []
test_list = []
for i in range(0,train.shape[2]):
    train_list.append(train[:,:,i])
    test_list.append(test[:,:,i])

# Initialize model
print('Building Model')
srm = SRM(n_iter=10, features=25)

# Fit model to training data (run 1)
print('Training Model')
srm.fit(train_list)

# Test model on testing data to produce shared response
print('Testing Model')
shared_data = srm.transform(test_list)

avg_response = sum(shared_data)/len(shared_data)
human_bounds = np.cumsum(np.load(datadir + 'songs2Dur.npy'))[:-1]
human_bounds2 = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

nR = shared_data[0].shape[0]
nTR = shared_data[0].shape[1]
nSubj = len(shared_data)

ev = brainiak.eventseg.event.EventSegment(16)
ev.fit(avg_response.T)

bounds = np.where(np.diff(np.argmax(ev.segments_[0], axis=1)))[0]

plt.figure(figsize=(10,10))
plt.imshow(np.corrcoef(avg_response.T))
plt.colorbar()
ax = plt.gca()
bounds_aug = np.concatenate(([0],bounds,[nTR]))
for i in range(len(bounds_aug)-1):
    rect = patches.Rectangle((bounds_aug[i],bounds_aug[i]),bounds_aug[i+1]-bounds_aug[i],bounds_aug[i+1]-bounds_aug[i],linewidth=3,edgecolor='w',facecolor='none')
    ax.add_patch(rect)

for i in range(len(human_bounds2)-1):
    rect = patches.Rectangle((human_bounds2[i],human_bounds2[i]),human_bounds2[i+1]-human_bounds2[i],human_bounds2[i+1]-human_bounds2[i],linewidth=3,edgecolor='k',facecolor='none')
    ax.add_patch(rect)

plt.title('HMM Fit to vmPFC',fontsize=18,fontweight='bold')
plt.xlabel('TRs',fontsize=18,fontweight='bold')
plt.ylabel('TRs',fontsize=18,fontweight='bold')

   
