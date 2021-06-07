import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

datadir = '/tigress/jamalw/MES/prototype/link/scripts/hmm_K_sweep_paper_results/principled/'

smooth_data_fn = 'smooth_wva_split_merge_01_lprec_rA1_bil_PHC_bil_mPFC'
smooth_data = np.load(datadir + smooth_data_fn + '_auto.npy')
roi_names = ['rA1', 'lprec', 'bil PHC', 'mPFC']

roi_data_mean = np.zeros((smooth_data.shape[1],smooth_data.shape[0]))

for i in range(roi_data_mean.shape[0]):
    roi_data_mean[i,:] = smooth_data[:,i,:].mean(axis=1) 


durs_run1 = np.array([225,90,180,135,90,180,135,90,180,135,225,90,225,225,180,135])

durs_run1_new = durs_run1[:,np.newaxis]

fairK = np.array((3,5,9,15,20,25,30,35,40,45))

event_lengths = durs_run1_new/fairK

unique_event_lengths = np.unique(event_lengths)
x = unique_event_lengths.ravel()

roi_min = np.zeros((roi_data_mean.shape[0],len(unique_event_lengths)))
roi_max = np.zeros((roi_data_mean.shape[0],len(unique_event_lengths)))

nBoot = 1000

# compute 95% confidence intervals for each bootstrap sample
for r in range(roi_data_mean.shape[0]):
    for i in range(len(unique_event_lengths)):
        roi_sorted = np.sort(smooth_data[i,r,:])
        roi_min[r,i] = roi_sorted[int(np.round(nBoot*0.025))]
        roi_max[r,i] = roi_sorted[int(np.round(nBoot*0.975))]          

plt.figure(figsize=(10,5))

for p in range(roi_data_mean.shape[0]):
    plt.plot(unique_event_lengths, roi_data_mean[p], label=roi_names[p],linewidth=3)
    plt.fill_between(unique_event_lengths, roi_min[p,:], roi_max[p,:], alpha=0.3)

plt.legend(fontsize=15)

event_lengths_str = ['2','','','3','','','','4','','5','','','','6','','','','','9','10','','12','15','18','20','25','27','30','36','45','60','75']

plt.xticks(unique_event_lengths,event_lengths_str,rotation=45,fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Event Length (s)', fontsize=18,fontweight='bold')
plt.ylabel('Model Fit', fontsize=18,fontweight='bold')
plt.title('Preferred Event Length', fontsize=18,fontweight='bold')
plt.tight_layout()

# compute pvals
prefix = 'A1_peaks_less_than_'
suffix = '_peaks'

# initialize variables for counting peaks less than A1 peak and storing corresponding p-values
roi_vs_a1_peaks = {}
roi_pvals = {}
for i in range(len(roi_names) - 1):
    roi_vs_a1_peaks['A1_peaks_less_than_' + roi_names[i+1] + '_peaks'] = np.zeros((1000))
    roi_pvals['pvals_A1_' + roi_names[i+1]] = np.zeros(())

roi_vs_a1_peaks_names = list(roi_vs_a1_peaks.keys())
roi_pvals_names = list(roi_pvals.keys())

# initialize variable for storing center of mass for each ROI
roi_com = {}
for c in range(len(roi_names)):    
    roi_com[roi_names[c] + '_com'] = np.zeros((1000))

roi_com_names = list(roi_com.keys())

x = np.arange(1,len(unique_event_lengths)+1)

# count A1 peaks less than ROI r peaks
for i in range(1000):
    peak_holder = np.zeros((len(roi_names)))
    for r1 in range(len(roi_names)):
        # compute peak for roi r as weighted average of wva and event length
        peak_holder[r1] = np.sum(x*smooth_data[:,r1,i])/np.sum(smooth_data[:,r1,i])
        # store center of mass in roi r at bootstrap #n
        roi_com[roi_com_names[r1]][i] = peak_holder[r1] 
    # store boolean for A1 peak less than ROI r peak
    for r2 in range(len(roi_names) - 1):
        roi_vs_a1_peaks[roi_vs_a1_peaks_names[r2]][i] = peak_holder[0] < peak_holder[r2+1]

# compute pvals for A1 peaks less than ROI r peak
for p in range(len(roi_pvals_names)):
    roi_pvals[roi_pvals_names[p]] = 1-np.sum(roi_vs_a1_peaks[roi_vs_a1_peaks_names[p]])/len(roi_vs_a1_peaks[roi_vs_a1_peaks_names[p]])


# compute rois preferred event length in seconds (max wva across bootstraps) and center of mass in seconds
pref_event_length_sec = {}
roi_com_mean = {}
for p in range(len(roi_data_mean)):
    pref_event_length_sec[roi_names[p]] = unique_event_lengths[np.argmax(roi_data_mean[p])]
    roi_com_mean[roi_com_names[p]] = np.mean(roi_com[roi_com_names[p]])


plt.savefig('hmm_K_sweep_paper_results/principled/' + smooth_data_fn + '_auto.png')


