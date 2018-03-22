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
train = np.nan_to_num(stats.zscore(np.load(datadir + 'vmPFC_point1_run1_n25.npy'),axis=1,ddof=1))
test = np.nan_to_num(stats.zscore(np.load(datadir + 'vmPFC_point1_run2_n25.npy'),axis=1,ddof=1))


# Convert data into lists where each element is voxels by samples
train_list = []
test_list = []
for i in range(0,train.shape[2]):
    #train_list.append(np.hstack((train[:,:,i],test[:,:,i])))
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

plt.imshow(np.corrcoef(avg_response[:,1480:1614].T))
plt.title("SRM Train Run 1 Test Run 2",fontweight='bold',fontsize=18)
