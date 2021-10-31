import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from save_nifti import save_nifti

datadir = '/tigress/jamalw/MES/prototype/link/scripts/hmm_K_sweep_paper_results/Schaefer300/DMN_no_srm/'

smooth = np.load(datadir + 'smooth_DMN_parcels_auto_independent_bandwidths.npy')

mask_img = nib.load('/tigress/jamalw/MES/data/mask_nonan.nii')

parcelNum = 300

parcels = nib.load("/tigress/jamalw/MES/data/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_" + str(parcelNum) + "Parcels_17Networks_order_FSLMNI152_2mm.nii.gz").get_data()

parcels_idx = np.concatenate((np.array([36,37,41]),np.arange(111,122),np.array([186,189]),np.arange(271,283)))

ev = np.load('/tigress/jamalw/MES/prototype/link/scripts/hmm_K_sweep_paper_results/principled/unique_event_lengths.npy')

midpt_brain = np.zeros_like(mask_img.get_data(),dtype=float)
peakfit_brain = np.zeros_like(mask_img.get_data(),dtype=float)

# Find the range of event lengths (in sec) that are > 95% of the maximum
# and report the midpoint
midpt = np.zeros((28,1000))

for i in range(28):
    indices = np.where((mask_img.get_data() > 0) & (parcels == parcels_idx[i]))
    for j in range(1000):
        # Find the first and last x value where smooth is within 5% of the peak
        peak_interval = np.where(smooth[:,i,j] > 0.95*smooth[:,i,j].max())[0][[0,-1]]
        
        # Average the event durations (in sec) for the first and last x values
        midpt[i,j] = ev[peak_interval].mean()

    midpt_brain[indices] = midpt[i,j]
    peakfit_brain[indices] = np.max(smooth.max(0)[i])

save_nifti(midpt_brain, mask_img.affine, datadir + 'preferred_event_lengths_DMN')
save_nifti(peakfit_brain, mask_img.affine, datadir + 'peak_fit_DMN')

## Plot peak wva fit for each ROI
#print(smooth.max(0))
#
#plt.savefig(datadir + 'peakfit_aud_prec_mpfc_ag_ant_hipp_post_hipp_independent_bandwidths.png')
#
