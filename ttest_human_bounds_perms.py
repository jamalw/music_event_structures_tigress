import numpy as np
import nibabel as nib
from scipy import stats
import glob

datadir = '/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_wva_shuffle_event_lengths/'

mask = nib.load('/tigress/jamalw/MES/data/mask_nonan.nii.gz')
mask_reshaped = np.reshape(mask.get_data(),(91*109*91)) != 0

affine = mask.affine

tmap_final1D = np.zeros((len(mask_reshaped)))
zmap_final1D = np.zeros((len(mask_reshaped)))
pmap_final1D = np.zeros((len(mask_reshaped)))
qmap_final1D = np.zeros((len(mask_reshaped)))

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

all_songs3D = np.zeros((91,109,91,len(songs)))
all_songs1D = np.zeros((218477,1001,len(songs)))

def FDR_p(pvals):
    # Port of AFNI mri_fdrize.c
    assert np.all(pvals>=0) and np.all(pvals<=1)
    pvals[pvals < np.finfo(np.float_).eps] = np.finfo(np.float_).eps
    pvals[pvals == 1] = 1-np.finfo(np.float_).eps
    n = pvals.shape[0]

    qvals = np.zeros((n))
    sorted_ind = np.argsort(pvals)
    sorted_pvals = pvals[sorted_ind]
    qmin = 1.0
    for i in range(n-1,-1,-1):
        qval = (n * sorted_pvals[i])/(i+1)
        if qval > qmin:
            qval = qmin
        else:
            qmin = qval
        qvals[sorted_ind[i]] = qval

    # Estimate number of true positives m1 and adjust q
    if n >= 233:
        phist = np.histogram(pvals, bins=20, range=(0, 1))[0]
        sorted_phist = np.sort(phist[3:19])
        if np.sum(sorted_phist) >= 160:
            median4 = n - 20*np.dot(np.array([1, 2, 2, 1]), sorted_phist[6:10])/6
            median6 = n - 20*np.dot(np.array([1, 2, 2, 2, 2, 1]), sorted_phist[5:11])/10
            m1 = min(median4, median6)

            qfac = (n - m1)/n
            if qfac < 0.5:
                qfac = 0.25 + qfac**2
            qvals *= qfac

    return qvals

for i in range(len(songs)):
    fn_z = glob.glob(datadir + songs[i] + '/perms/full_brain/avg_perms.npy')
    data = np.load(fn_z[0])
    all_songs3D[:,:,:,i] = data[:,:,:,0]
    all_songs1D[:,:,i] = data[mask.get_data() != 0]

zmap_final3D = np.mean(all_songs3D,axis=3)

tmap_final4D = np.zeros((91,109,91,1001))
pmap_final4D = np.zeros((91,109,91,1001))
qmap_final4D = np.zeros((91,109,91,1001))
zmap_final4D = np.zeros((91,109,91,1001))

for i in range(1001):
    tmap1D = np.zeros((len(all_songs1D[:,:,0])))
    pmap1D = np.zeros((len(all_songs1D[:,:,0])))
    qmap1D = np.zeros((len(all_songs1D[:,:,0])))
    zmap1D = np.zeros((len(all_songs1D[:,:,0])))

    for j in range(len(all_songs1D[:,:,0])):
        tmap1D[j],pmap1D[j] = stats.ttest_1samp(all_songs1D[j,i,:],0,axis=0)
        if all_songs1D[j,:].mean() > 0:
            pmap1D[j] = pmap1D[j]/2
        else:
            pmap1D[j] = 1-pmap1D[j]/2

    qmap1D = FDR_p(pmap1D)

    # Fit data back into whole brain
    tmap_final1D[mask_reshaped==1] = tmap1D
    tmap_final3D = np.reshape(tmap_final1D,(91,109,91))
    tmap_final4D[:,:,:,i] = tmap_final3D    

    zmap_final1D[mask_reshaped==1] = scipy.stats.zscore(tmap1D)
    zmap_final3D = np.reshape(zmap_final1D,(91,109,91))
    zmap_final4D[:,:,:,i] = zmap_final3D

    pmap_final1D[mask_reshaped==1] = pmap1D
    pmap_final3D = np.reshape(pmap_final1D,(91,109,91))
    pmap_final4D[:,:,:,i] = pmap_final3D

    qmap_final1D[mask_reshaped==1] = qmap1D
    qmap_final3D = np.reshape(qmap_final1D,(91,109,91))
    qmap_final4D[:,:,:,i] = qmap_final3D

# save data
maxval = np.max(tmap_final4D[:,:,:,0])
minval = np.min(tmap_final4D[:,:,:,0])
img = nib.Nifti1Image(tmap_final4D[:,:,:,0], affine=affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + '/ttest_results/tstats_map_both_runs.nii.gz')

maxval = np.max(pmap_final4D[:,:,:,0])
minval = np.min(pmap_final4D[:,:,:,0])
img = nib.Nifti1Image(pmap_final4D[:,:,:,0], affine=affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + '/ttest_results/pstats_map_both_runs.nii.gz')

np.save(datadir + '/ttest_results/pstats_map_both_runs_w_perms',pmap_final4D)
np.save(datadir + '/ttest_results/tstats_map_both_runs_w_perms',tmap_final4D)
np.save(datadir + '/ttest_results/zstats_map_both_runs_w_perms', zmap_final4D)

maxval = np.max(qmap_final4D[:,:,:,0])
minval = np.min(qmap_final4D[:,:,:,0])
img = nib.Nifti1Image(qmap_final4D[:,:,:,0], affine=affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + '/ttest_results/qstats_map_both_runs.nii.gz')

maxval = np.max(zmap_final3D)
minval = np.min(zmap_final3D)
img = nib.Nifti1Image(zmap_final3D, affine=affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + '/ttest_results/zstats_map_both_runs.nii.gz')

