import numpy as np
import matplotlib.pyplot as plt

datadir = '/tigress/jamalw/MES/prototype/link/scripts/hmm_K_sweep_paper_results/principled/'
smooth = np.load(datadir + 'smooth_wva_split_merge_01_bil_A1_no_srm_ver2_bil_precuneus_no_srm_ver2_bil_mPFC_no_srm_ver2_bil_AG_no_srm_ver2_auto.npy')
ev = np.load(datadir + 'unique_event_lengths.npy')

# Find the range of event lengths (in sec) that are > 95% of the maximum
# and report the midpoint
midpt = np.zeros((4,1000))
for i in range(4):
    for j in range(1000):
        # Find the first and last x value where smooth is within 5% of the peak
        peak_interval = np.where(smooth[:,i,j] > 0.95*smooth[:,i,j].max())[0][[0,-1]]

        # Average the event durations (in sec) for the first and last x values
        midpt[i,j] = ev[peak_interval].mean()

# Print p values for mPFC less than all ROIs
#print((midpt[2,:]<=midpt).mean(1))

# Print p values for A1 greater than all other ROIs
print((midpt[0,:]>=midpt).mean(1))

# Print mean preferred event length for each ROI
print(midpt.mean(1))
# Plot bootstrap dist of preferred event lengths
plt.figure(1)
plt.violinplot(midpt.T,showextrema=False);

plt.title('preferred event lengths',fontsize=20)
plt.ylabel('event length (s)',fontsize=17) 
plt.xticks(np.arange(1,5),['bil aud', 'bil prec', 'bil mPFC', 'bil AG'],fontsize=15)

#plt.savefig(datadir + 'peakdiff.png')

# Plot peak wva fit for each ROI
print(smooth.max(0))
print((smooth.max(0)[3]<=smooth.max(0)).mean(1))
plt.figure(2)
plt.violinplot(smooth.max(0).T,showextrema=False)

plt.title('peak fit',fontsize=20)
plt.ylabel('Within vs. Across (r)',fontsize=17) 
plt.xticks(np.arange(1,5),['bil aud', 'bil prec', 'bil mPFC', 'bil AG'],fontsize=15)

plt.savefig(datadir + 'peakfit.png')

plt.show()
