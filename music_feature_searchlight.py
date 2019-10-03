import numpy as np
import brainiak.eventseg.event
from scipy.stats import norm,zscore,pearsonr,stats
from nilearn.image import load_img
import sys
from brainiak.funcalign.srm import SRM
import nibabel as nib
import os
from scipy.spatial import distance
from sklearn import linear_model
from scipy.spatial.distance import squareform, pdist
import logging
from scipy.fftpack import fft, ifft

logger = logging.getLogger(__name__)

datadir = '/jukebox/norman/jamalw/MES/'

subjs = ['MES_022817_0','MES_030217_0','MES_032117_1','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0','MES_041417_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

# run 1 times
#song_bounds = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973, 2198,2377,2511]) 

#songs = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

#song_features = np.load(datadir + 'prototype/link/scripts/data/searchlight_input/mfccRun1_hrf.npy')

# run 2 times
song_bounds = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

song_features = np.load(datadir + 'prototype/link/scripts/data/searchlight_input/chromaRun2_hrf.npy')

song_idx = int(sys.argv[1])
srm_k = 30
hrf = 5

mask_img = load_img(datadir + 'data/mask_nonan.nii.gz')
mask = mask_img.get_data()
mask_reshape = np.reshape(mask,(91*109*91))

def _check_timeseries_input(data):

    """Checks response time series input data (e.g., for ISC analysis)
    Input data should be a n_TRs by n_voxels by n_subjects ndarray
    (e.g., brainiak.image.MaskedMultiSubjectData) or a list where each
    item is a n_TRs by n_voxels ndarray for a given subject. Multiple
    input ndarrays must be the same shape. If a 2D array is supplied,
    the last dimension is assumed to correspond to subjects. This
    function is generally intended to be used internally by other
    functions module (e.g., isc, isfc in brainiak.isc).
    Parameters
    ----------
    data : ndarray or list
        Time series data
    Returns
    -------
    data : ndarray
        Input time series data with standardized structure
    n_TRs : int
        Number of time points (TRs)
    n_voxels : int
        Number of voxels (or ROIs)
    n_subjects : int
        Number of subjects
    """

    # Convert list input to 3d and check shapes
    if type(data) == list:
        data_shape = data[0].shape
        for i, d in enumerate(data):
            if d.shape != data_shape:
                raise ValueError("All ndarrays in input list "
                                 "must be the same shape!")
            if d.ndim == 1:
                data[i] = d[:, np.newaxis]
        data = np.dstack(data)

    # Convert input ndarray to 3d and check shape
    elif isinstance(data, np.ndarray):
        if data.ndim == 2:
            data = data[:, np.newaxis, :]
        elif data.ndim == 3:
            pass
        else:
            raise ValueError("Input ndarray should have 2 "
                             "or 3 dimensions (got {0})!".format(data.ndim))

    # Infer subjects, TRs, voxels and log for user to check
    n_TRs, n_voxels, n_subjects = data.shape
    logger.info("Assuming {0} subjects with {1} time points "
                "and {2} voxel(s) or ROI(s) for ISC analysis.".format(
                    n_subjects, n_TRs, n_voxels))

    return data, n_TRs, n_voxels, n_subjects

def phase_randomize(data, voxelwise=False, random_state=None):
    """Randomize phase of time series across subjects
    For each subject, apply Fourier transform to voxel time series
    and then randomly shift the phase of each frequency before inverting
    back into the time domain. This yields time series with the same power
    spectrum (and thus the same autocorrelation) as the original time series
    but will remove any meaningful temporal relationships among time series
    across subjects. By default (voxelwise=False), the same phase shift is
    applied across all voxels; however if voxelwise=True, different random
    phase shifts are applied to each voxel. The typical input is a time by
    voxels by subjects ndarray. The first dimension is assumed to be the
    time dimension and will be phase randomized. If a 2-dimensional ndarray
    is provided, the last dimension is assumed to be subjects, and different
    phase randomizations will be applied to each subject.
    The implementation is based on the work in [Lerner2011]_ and
    [Simony2016]_.
    Parameters
    ----------
    data : ndarray (n_TRs x n_voxels x n_subjects)
        Data to be phase randomized (per subject)
    voxelwise : bool, default: False
        Apply same (False) or different (True) randomizations across voxels
    random_state : RandomState or an int seed (0 by default)
        A random number generator instance to define the state of the
        random permutations generator.
    Returns
    ----------
    shifted_data : ndarray (n_TRs x n_voxels x n_subjects)
        Phase-randomized time series
    """

    # Check if input is 2-dimensional
    data_ndim = data.ndim

    # Get basic shape of data
    data, n_TRs, n_voxels, n_subjects = _check_timeseries_input(data)

    # Random seed to be deterministically re-randomized at each iteration
    if isinstance(random_state, np.random.RandomState):
        prng = random_state
    else:
        prng = np.random.RandomState(random_state)

    # Get randomized phase shifts
    if n_TRs % 2 == 0:
        # Why are we indexing from 1 not zero here? n_TRs / -1 long?
        pos_freq = np.arange(1, data.shape[0] // 2)
        neg_freq = np.arange(data.shape[0] - 1, data.shape[0] // 2, -1)
    else:
        pos_freq = np.arange(1, (data.shape[0] - 1) // 2 + 1)
        neg_freq = np.arange(data.shape[0] - 1,
                             (data.shape[0] - 1) // 2, -1)

    if not voxelwise:
        phase_shifts = (prng.rand(len(pos_freq), 1, n_subjects)
                        * 2 * np.math.pi)
    else:
        phase_shifts = (prng.rand(len(pos_freq), n_voxels, n_subjects)
                        * 2 * np.math.pi)

    # Fast Fourier transform along time dimension of data
    fft_data = fft(data, axis=0)

    # Shift pos and neg frequencies symmetrically, to keep signal real
    fft_data[pos_freq, :, :] *= np.exp(1j * phase_shifts)
    fft_data[neg_freq, :, :] *= np.exp(-1j * phase_shifts)

    # Inverse FFT to put data back in time domain
    shifted_data = np.real(ifft(fft_data, axis=0))

    # Go back to 2-dimensions if input was 2-dimensional
    if data_ndim == 2:
        shifted_data = shifted_data[:, 0, :]

    return shifted_data


def searchlight(coords,song_features,mask,subjs,song_idx,song_bounds,srm_k,hrf):
    
    """run searchlight 

       Create searchlight object and perform voxel function at each searchlight location
    
       Parameters
       ----------
       data1  : voxel by time ndarray (2D); leftout subject run 1
       data2  : voxel by time ndarray (2D); average of others run 1
       data3  : voxel by time ndarray (2D); leftout subject run 2
       data4  : voxel by time ndarray (2D); average of others run 2
       coords : voxel by xyz ndarray (2D, Vx3)
       K      : # of events for HMM (scalar)
       
       Returns
       -------
       3D data: brain (or ROI) filled with searchlight function scores (3D)

    """

    stride = 5
    radius = 5
    min_vox = srm_k
    nPerm = 1000
    SL_allvox = []
    SL_results = []
    datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_input/'
    for x in range(0,np.max(coords, axis=0)[0]+stride,stride):
        for y in range(0,np.max(coords, axis=0)[1]+stride,stride):
           for z in range(0,np.max(coords, axis=0)[2]+stride,stride):
               if not os.path.isfile(datadir + subjs[0] + '/' + str(x) + '_' + str(y) + '_' + str(z) + '.npy'):
                   continue
               D = distance.cdist(coords,np.array([x,y,z]).reshape((1,3)))[:,0]
               SL_vox = D <= radius
               data = []
               for i in range(len(subjs)):
                   subj_data = np.load(datadir + subjs[i] + '/' + str(x) + '_' + str(y) + '_' + str(z) + '.npy')
                   subj_regs = np.genfromtxt(datadir + subjs[i] + '/EPI_mcf1.par')
                   motion = subj_regs.T
                   regr = linear_model.LinearRegression()
                   regr.fit(motion[:,0:2511].T,subj_data[:,:,0].T)
                   subj_data1 = subj_data[:,:,0] - np.dot(regr.coef_, motion[:,0:2511]) - regr.intercept_[:, np.newaxis]
                   data.append(np.nan_to_num(stats.zscore(subj_data1,axis=1,ddof=1)))
               for i in range(len(subjs)):
                   subj_data = np.load(datadir + subjs[i] + '/' + str(x) + '_' + str(y) + '_' + str(z) + '.npy')
                   subj_regs = np.genfromtxt(datadir + subjs[i] + '/EPI_mcf2.par')
                   motion = subj_regs.T
                   regr = linear_model.LinearRegression()
                   regr.fit(motion[:,0:2511].T,subj_data[:,:,1].T)
                   subj_data2 = subj_data[:,:,1] - np.dot(regr.coef_, motion[:,0:2511]) - regr.intercept_[:, np.newaxis]
                   data.append(np.nan_to_num(stats.zscore(subj_data2,axis=1,ddof=1))) 
               print("Running Searchlight")
               # only run function on searchlights with voxels greater than or equal to min_vox
               if data[0].shape[0] >= min_vox: 
                   SL_match = RSA(data,song_features,song_idx,song_bounds,srm_k,hrf)
                   SL_results.append(SL_match)
                   SL_allvox.append(np.array(np.nonzero(SL_vox)[0])) 
    voxmean = np.zeros((coords.shape[0], nPerm+1))
    vox_SLcount = np.zeros(coords.shape[0])
    for sl in range(len(SL_results)):
       voxmean[SL_allvox[sl],:] += SL_results[sl]
       vox_SLcount[SL_allvox[sl]] += 1
    voxmean = voxmean / vox_SLcount[:,np.newaxis]
    vox_z = np.zeros((coords.shape[0], nPerm+1))
    for p in range(nPerm+1):
        vox_z[:,p] = (voxmean[:,p] - np.mean(voxmean[:,1:],axis=1))/np.std(voxmean[:,1:],axis=1) 
    return vox_z,voxmean

def RSA(X,song_features,song_idx,song_bounds,srm_k,hrf):
    
    """compute RSA for music features and neural data
  
       compute similarity matrices for music and neural features then compute RSA between the two             

       Parameters
       ----------
       A: voxel by time ndarray (2D)
       B: voxel by time ndarray (2D)
       C: voxel by time ndarray (2D)
       D: voxel by time ndarray (2D)
       K: # of events for HMM (scalar)
 
       Returns
       -------
       z: z-score after permuting music feature vectors      

    """
    
    nPerm = 1000
    run1 = [X[i] for i in np.arange(0, int(len(X)/2))]
    run2 = [X[i] for i in np.arange(int(len(X)/2), len(X))]
    print('Building Model')
    srm = SRM(n_iter=10, features=srm_k)   
    print('Training Model')
    srm.fit(run1)
    print('Testing Model')
    shared_data = srm.transform(run2)
    shared_data = stats.zscore(np.dstack(shared_data),axis=1,ddof=1)
    
    data = np.mean(shared_data[:,song_bounds[song_idx]:song_bounds[song_idx + 1]],axis=2)
    
    song_features = song_features[:,song_bounds[song_idx]:song_bounds[song_idx + 1]]

    rsa_scores = np.zeros(nPerm+1)
    perm_features = song_features.copy()
 
    songCorr = np.corrcoef(song_features.T)
    songVec = squareform(songCorr, checks=False)
    voxCorr = np.corrcoef(data.T)
    voxVec = squareform(voxCorr,checks=False)
 
    for p in range(nPerm+1):
        spearmanCorr = stats.spearmanr(voxVec,songVec,nan_policy='omit')
        rsa_scores[p] = spearmanCorr[0]
        song_features = phase_randomize(song_features.T, voxelwise=False, random_state=p).T
        songCorr = np.corrcoef(song_features.T)
        songVec = squareform(songCorr, checks=False)

    return rsa_scores


# initialize data stores
global_outputs_all = np.zeros((91,109,91))
results3d = np.zeros((91,109,91))
results3d_real = np.zeros((91,109,91))
results3d_perms = np.zeros((91,109,91,1001))
# create coords matrix
x,y,z = np.mgrid[[slice(dm) for dm in tuple((91,109,91))]]
x = np.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2]))
y = np.reshape(y,(y.shape[0]*y.shape[1]*y.shape[2]))
z = np.reshape(z,(z.shape[0]*z.shape[1]*z.shape[2]))
coords = np.vstack((x,y,z)).T 
coords_mask = coords[mask_reshape>0]
print('Running Distribute...')
voxmean,real_sl_scores = searchlight(coords_mask,song_features,mask,subjs,song_idx,song_bounds,srm_k,hrf) 
results3d[mask>0] = voxmean[:,0]
results3d_real[mask>0] = real_sl_scores[:,0]
for j in range(voxmean.shape[1]):
    results3d_perms[mask>0,j] = voxmean[:,j]
 
print('Saving data to Searchlight Folder')
print(songs[song_idx])
np.save('/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/music_features/' + songs[song_idx] +'/chroma/raw/globals_raw_srm_k_' + str(srm_k) + '_fit_run1', results3d_real)
np.save('/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/music_features/' + songs[song_idx] +'/chroma/zscores/globals_z_srm_k' + str(srm_k) + '_fit_run1', results3d)
np.save('/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/music_features/' + songs[song_idx] +'/chroma/perms/globals_z_srm_k' + str(srm_k) + '_fit_run1', results3d_perms)


