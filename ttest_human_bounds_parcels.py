import numpy as np
import nibabel as nib
from scipy import stats
import glob

datadir = '/tigress/jamalw/MES/prototype/link/scripts/data/searchlight_output/parcels/Schaefer100/'

parcels = nib.load("/tigress/jamalw/MES/data/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz").get_data()

mask = nib.load('/tigress/jamalw/MES/data/mask_nonan.nii')

# Schaefer 100 DMN and Auditory
#parcel_idx = np.concatenate([np.arange(10,13),np.arange(38,50),np.arange(61,64),np.arange(89,98)])

# Schaefer 100 All
parcel_idx = np.arange(1,101)

#Schaefer 200 DMN and Auditory
#parcel_idx = np.concatenate([np.arange(21,27),np.arange(75,99),np.arange(124,130),np.arange(184,197)])

x = []
y = []
z = []

x_single = []
y_single = []
z_single = []

for i in range(len(parcel_idx)):
    mask_x = np.where((mask.get_data() > 0) & (parcels == parcel_idx[i]))[0]
    mask_y = np.where((mask.get_data() > 0) & (parcels == parcel_idx[i]))[1]
    mask_z = np.where((mask.get_data() > 0) & (parcels == parcel_idx[i]))[2]
    x.append(mask_x)
    y.append(mask_y)
    z.append(mask_z)
    # take first set of coordinates from mask for each parcel 
    x_single.append(mask_x[0])
    y_single.append(mask_y[0])
    z_single.append(mask_z[0])

x_stack = np.hstack(x)
y_stack = np.hstack(y)
z_stack = np.hstack(z)

indices = np.array((x_stack,y_stack,z_stack)) 

x_single_stack = np.hstack(x_single)
y_single_stack = np.hstack(y_single)
z_single_stack = np.hstack(z_single)

single_indices = np.array((x_single_stack,y_single_stack,z_single_stack)) 

mask_reshaped = np.reshape(mask.get_data(),(91*109*91))
tmap_final3D = np.zeros_like(mask.get_data(),dtype=float)
pmap_final3D = np.zeros_like(mask.get_data(),dtype=float)
qmap_final3D = np.zeros_like(mask.get_data(),dtype=float)

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

all_songs3D = np.zeros((91,109,91,len(songs)))
all_songs1D = np.zeros((len(single_indices[0]),len(songs)))

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
    data = nib.load(datadir + songs[i] + '/globals_avg_wva_both_z_runs.nii.gz').get_data()
    all_songs3D[indices[0],indices[1],indices[2],i] = data[indices[0],indices[1],indices[2]]
    all_songs1D[:,i] = data[single_indices[0],single_indices[1],single_indices[2]]

zmap_final3D = np.mean(all_songs3D,axis=3)

tmap1D = np.zeros((len(all_songs1D[:,0])))
pmap1D = np.zeros((len(all_songs1D[:,0])))
qmap1D = np.zeros((len(all_songs1D[:,0])))

for j in range(len(all_songs1D[:,0])):
	tmap1D[j],pmap1D[j] = stats.ttest_1samp(all_songs1D[j,:],0,axis=0)
	if all_songs1D[j,:].mean() > 0:
		pmap1D[j] = pmap1D[j]/2
	else:
		pmap1D[j] = 1-pmap1D[j]/2


qmap1D = FDR_p(pmap1D)

# Fit data back into whole brain
for i in range(len(parcel_idx)):
    tmap_final3D[parcels==parcel_idx[i]] = tmap1D[i]
    pmap_final3D[parcels==parcel_idx[i]] = pmap1D[i]
    qmap_final3D[parcels==parcel_idx[i]] = qmap1D[i]


# save data
maxval = np.max(tmap_final3D)
minval = np.min(tmap_final3D)
img = nib.Nifti1Image(tmap_final3D, affine=mask.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + 'ttest_results/tstats_map_both_runs_srm_v1_all_100.nii.gz')

maxval = np.max(pmap_final3D)
minval = np.min(pmap_final3D)
img = nib.Nifti1Image(pmap_final3D, affine=mask.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + 'ttest_results/pstats_map_both_runs_srm_v1_all_100.nii.gz')

maxval = np.max(qmap_final3D)
minval = np.min(qmap_final3D)
img = nib.Nifti1Image(qmap_final3D, affine=mask.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + 'ttest_results/qstats_map_both_runs_srm_v1_all_100.nii.gz')

maxval = np.max(zmap_final3D)
minval = np.min(zmap_final3D)
img = nib.Nifti1Image(zmap_final3D, affine=mask.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + 'ttest_results/zstats_map_both_runs_srm_v1_all_100.nii.gz')
