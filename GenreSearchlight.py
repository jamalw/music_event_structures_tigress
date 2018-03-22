import numpy as np
from nilearn.image import load_img
import sys
from brainiak.brainiak.searchlight.searchlight import Searchlight

subj = sys.argv[1]

datadir = '/Users/jamalw/Desktop/PNI/music_event_structures/'

data1 = load_img(datadir + 'subjects/' + subj + '/trans_filtered_func_data1.nii')
data2 = load_img(datadir + 'subjects/' + subj + '/trans_filtered_func_data2.nii')
mask_img = load_img(datadir + 'a1plus_2mm.nii')

data1 = data1.get_data()[...,:-1]
data2 = data2.get_data()
mask_img = mask_img.get_data()

def corr2_coeff(A,B):
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))

sl = Searchlight(sl_rad=1,max_blk_edge=5)
sl.distribute([data1,data2],mask_img)
print('Running Searchlight...')
global_outputs = sl.run_searchlight(corr2_coeff(data1,data2))
