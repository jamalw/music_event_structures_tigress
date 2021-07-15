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



vox_features = phase_randomize(perm_features.T, voxelwise=False, random_state=p).T
    


