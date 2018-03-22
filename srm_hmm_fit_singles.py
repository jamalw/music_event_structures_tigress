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

songs = ['St Pauls Suite', 'I Love Music', 'Moonlight Sonata', 'Change of the Gaurd','Waltz of Flowers','The Bird', 'Island', 'Allegro Moderato', 'Finlandia', 'Early Summer', 'Capriccio Espagnole', 'Symphony Fantastique', 'Boogie Stop Shuffle', 'My Favorite Things', 'Blue Monk','All Blues']

# Load in data
train = np.nan_to_num(stats.zscore(np.load(datadir + 'A1_run1_n25.npy'),axis=1,ddof=1))
test = np.nan_to_num(stats.zscore(np.load(datadir + 'A1_run2_n25.npy'),axis=1,ddof=1))

# Convert data into lists where each element is voxels by samples
train_list = []
test_list = []
for i in range(0,train.shape[2]):
    train_list.append(train[:,:,i])
    test_list.append(test[:,:,i])

# Initialize model
print('Building Model')
srm = SRM(n_iter=10, features=10)

# Fit model to training data (run 1)
print('Training Model')
srm.fit(train_list)

# Test model on testing data to produce shared response
print('Testing Model')
shared_data = srm.transform(test_list)

avg_response = sum(shared_data)/len(shared_data)
human_bounds = np.cumsum(np.load(datadir + 'songs2Dur.npy'))[:-1]
human_bounds2 = np.array([0, 11,  22,  32,  57,  78, 101, 110, 123, 146, 167, 202, 212,225])

nR = shared_data[0].shape[0]
nTR = shared_data[0][:,1839:2063].shape[1]
nSubj = len(shared_data)

ev = brainiak.eventseg.event.EventSegment(14)
ev.fit(avg_response[:,1839:2063].T)

bounds = np.where(np.diff(np.argmax(ev.segments_[0], axis=1)))[0]

plt.figure(figsize=(10,10))
plt.imshow(np.corrcoef(avg_response[:,1839:2063].T))
plt.colorbar()
ax = plt.gca()
bounds_aug = np.concatenate(([0],bounds,[nTR]))
for i in range(len(bounds_aug)-1):
    rect1 = patches.Rectangle((bounds_aug[i],bounds_aug[i]),bounds_aug[i+1]-bounds_aug[i],bounds_aug[i+1]-bounds_aug[i],linewidth=3,edgecolor='w',facecolor='none',label='Model Fit')
    ax.add_patch(rect1)

for i in range(len(human_bounds2)-1):
    rect2 = patches.Rectangle((human_bounds2[i],human_bounds2[i]),human_bounds2[i+1]-human_bounds2[i],human_bounds2[i+1]-human_bounds2[i],linewidth=3,edgecolor='k',facecolor='none',label='Human Annotations')
    ax.add_patch(rect2)

plt.title('HMM Fit to AG',fontsize=18,fontweight='bold')
plt.xlabel('TRs',fontsize=18,fontweight='bold')
plt.ylabel('TRs',fontsize=18,fontweight='bold')
#plt.legend(handles=[rect1,rect2])

#for i in range(len(human_bounds2)-1):
#    plt.figure(i+1)
#    plt.imshow(np.corrcoef(avg_response[:,human_bounds2[i]:human_bounds2[i+1]].T))
#    plt.title(songs[i] + ' A1 SRM',fontsize=18)
#    plt.xlabel('TRs',fontsize=18,fontweight='bold')
#    plt.ylabel('TRs',fontsize=18,fontweight='bold')    
#    plt.savefig('plots/'+songs[i] + '_A1_SRM') 
