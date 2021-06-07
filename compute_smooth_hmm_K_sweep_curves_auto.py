import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from statsmodels.nonparametric.kernel_regression import KernelReg

datadir = '/tigress/jamalw/MES/prototype/link/scripts/hmm_K_sweep_paper_results/principled/'

roi_input_IDs = ['rA1', 'lprec', 'bil_PHC', 'bil_mPFC']
suffix = 'wva_split_merge_01'
save_char = '_'
save_fn = save_char.join(roi_input_IDs)

jobNum = 20
nBoots = 1000

# initialize variable number lists corresponding to each ROI
lists = [[] for _ in range(len(roi_input_IDs))]
ROI_data = []  

# loop over each list for each ROI and store WvA results 
for r in range(len(lists)):
    # load in each job (20 of which contain 50 bootstraps each which is 1000 boostraps total) for each ROI separately to be converted into one large matrix containing all bootstraps
    for j in range(jobNum):
        lists[r].append(np.load(datadir + roi_input_IDs[r] + '_' + suffix + '_' + str(j) + '.npy'))
    ROI_data.append(np.dstack(lists[r]))

sigma = '5'

durs_run1 = np.array([225,90,180,135,90,180,135,90,180,135,225,90,225,225,180,135])

durs_run1_new = durs_run1[:,np.newaxis]

fairK = np.array((3,5,9,15,20,25,30,35,40,45))

event_lengths = durs_run1_new/fairK

unique_event_lengths = np.unique(event_lengths)
x = event_lengths.ravel()

test_x = np.linspace(min(x), max(x), num=100)
smooth_wva = np.zeros((len(unique_event_lengths), len(ROI_data), nBoots))

for b in range(nBoots):
    # Optimize bandwidth
    opt_bw = 0
    for ROI in range(len(ROI_data)):
        y = ROI_data[ROI][:,:,b].ravel()
        KR = KernelReg(y,x,var_type='c')
        opt_bw += KR.bw/len(ROI_data)

    max_wva = np.zeros(len(ROI_data))
    for ROI in range(len(ROI_data)):
        y = ROI_data[ROI][:,:,b].ravel()
        KR = KernelReg(y,x,var_type='c', bw=opt_bw)
        max_wva[ROI] = np.argmax(KR.fit(test_x)[0])  # Find peak on fine grid
        smooth_wva[:, ROI, b] += KR.fit(unique_event_lengths)[0]

np.save(datadir + 'smooth_' + suffix + '_' + save_fn + '_auto', smooth_wva)


