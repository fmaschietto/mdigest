"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: fmaschietto, bcallen95"""

from mdigest.core.imports import *
from sklearn.neighbors import KDTree
from scipy.special import digamma
import numba

##############################################################################
################# Correlation related functions #############################
##############################################################################
def _extract_coords(u, selection='protein'):
    """
    Utility function to extract coordinates from an MDA.Universe given some atom selecton.

    Parameters
    ----------
    u: mda.Universe object,
        MDAnalysis universe
    selection: str,
        atom selection string

    Returns
    -------
    all_coords: np.ndarray
        coordinates
    """
    nfs = u.trajectory.n_frames
    subset = u.select_atoms(selection)
    all_coords = np.zeros(shape=(nfs, len(subset), 3))
    counter = 0
    for ts in u.trajectory:
        temp_pos = subset.atoms.positions
        all_coords[counter] = temp_pos
        counter += 1
    return all_coords


def coordinate_reshape(values):
    """
    Reshape values to shape (nframes, nfeatures*features_dimensions) where
         - ``nframes`` is the number of frames of the MD timeseries considered,
         - ``nfeatures`` is the number of features, i.e. number of atoms for which values are stored,
         - ``features_dimensions`` is the dimensionality of the features (i.e. 3 for xyz-coordinates, 4 for [sin(phi), cos(phi), sin(psi), cos(psi)] dihedral coordinate vector).

    Parameters
    ----------
    values: np.ndarray, shape (nframes, nfeatures, features_dimensions)
        values


    Returns
    -------
    values_: np.ndarray, shape (nframes, nfeatures * features_dimensions)
        reshaped values

    """

    nframes = values.shape[0]
    nfeatures = values.shape[1]
    features_dimension = values.shape[2]

    values_ = values.reshape((nframes, nfeatures * features_dimension))

    for i in range(0, nframes):
        ccount = 0
        for j in range(0, nfeatures):
            values_[i, ccount + 0] = values[i, j, 0]
            values_[i, ccount + 1] = values[i, j, 1]
            values_[i, ccount + 2] = values[i, j, 2]
            ccount += features_dimension
    ccount = 0

    return values_


def _compute_displacements(values):
    """
    Compute atomic displacements

    Parameters
    ----------
    values: np.ndarray, shape``nframes, nfeatures, features_dimensions``
        contains features coordinate vectors, for example xyz-coordinates or dihedral vectors.
         - ``nframes`` is the number of frames of the MD timeseries considered,
         - ``nfeatures`` is the number of features, i.e. number of atoms for which values are stored,
         - ``features_dimensions`` is the dimensionality of the features (i.e. 3 for xyz-coordinates, 4 for [sin(phi), cos(phi), sin(psi), cos(psi)] dihedral coordinate vector).

    Returns
    -------
    displacements: np.ndarray, shape (nframes, nfeatures * features_dimensions)
        displacements from mean position
    """

    nframes             = values.shape[0]
    nfeatures           = values.shape[1]
    features_dimensions = values.shape[2]

    values_ = values.reshape((nframes, nfeatures * features_dimensions))
    average = values_.mean(axis=0)

    displacements = np.zeros((nframes, nfeatures * features_dimensions))

    for i in range(0, features_dimensions * nfeatures):
        for j in range(0, nframes):
            displacements[j][i] = values_[j][i] - average[i]
    return displacements


def _compute_square_displacements(values):
    """
    Compute square displacements from features array.

    Parameters
    ----------
    values: np.ndarray,
        values of shapenframes, nfeatures, features_dimensions

    Returns
    -------
    displacements: np.ndarray, shape(nframes, nfeatures*1)
        displacements matrix
    """

    nframes             = values.shape[0]
    nfeatures           = values.shape[1]

    displacements = np.zeros((nframes, nfeatures))
    average = values.mean(axis=0)

    for j in range(0, nframes):
        displacements[j] = np.linalg.norm(values[j] - average, axis=1)
    return displacements


def compute_distance_matrix(values, features_dimensions):
    """
    Compute distance matrix using the average position over the entire lentgh of the given trajectory

    Parameters
    ----------
    values: np.ndarray,
        array of values of shape (nframes, nfeatures * features_dimensions)
    features_dimensions: int,
        dimensionality of features (3 if values contains xyz coordinates)

    Returns
    -------
    dist_matrix: np.ndarray, shape (nfeatures, nfeatures)
        distance matrix
    """

    from scipy.spatial import distance_matrix
    nfeatures = int(values.shape[1]/features_dimensions)
    values_avg = np.mean(values.view(), axis=0)
    values_cop = np.reshape(values_avg, [nfeatures, features_dimensions])
    dist_matrix = distance_matrix(values_cop, values_cop)
    return dist_matrix


def evaluate_covariance_matrix(values, center='square_disp'):
    """
    Compute the covariance matrix of features values.

    Parameters
    ----------
    values: np.ndarray, shape (nframes, nfeatures, features_dimensions)
        matrix of displacements from mean values
    center: str,
        -if 'mean' remove mean
        -if 'square_disp' compute square displacements of values array (default option)

    Returns
    -------
    covar_mat: np.ndarray, shape (nfeatures, nfeatures)
        covariance matrix
    """

    nframes             = values.shape[0]
    nfeatures           = values.shape[1]
    features_dimensions = values.shape[2]

    if center == 'mean':
        values_ = _compute_displacements(values)

    elif center == 'square_disp':
        values_ = _compute_square_displacements(values)


    else:
        values_ = values.reshape((nframes, nfeatures * features_dimensions))

    covar_mat = np.cov(values_, rowvar=False, bias=True)

    return covar_mat



def _mutual_information_gaussian_estimator(values, features_dimension, correction=False):
        """
        Computes linearized mutual information (LMI), based on Gaussian estimator.
            Unlike Pearson correlation, LMI detects orthogonal correlation but still does not capture non linear correlation.
            Allows for rapid performance with results that in most cases compare very well to those obtained using
            non-linear estimators of MI.

        Parameters
        ----------
        values: np.array,
            values should have shape (n_samples, n_features, feature_dimension)
        features_dimension: int,
            features shape(i.e. 3 for xyz coordinates)

        Returns
        -------
        mutual_information: np.ndarray, shape (nfeatures, nfeatures)
            linearized mutual information matrix
        """

        n_features = values.shape[1]

        if features_dimension == 1:
            print('@>: values array has shape {}'.format(values.shape))
            rho = np.corrcoef(values, rowvar=False)
            np.fill_diagonal(rho, 0)
            mutual_information = -0.5 * np.emath.log((1 - rho ** 2))
            if correction:
                mutual_information -= np.min(mutual_information)

        else:
            print('@>: values array has shape {}'.format(values.shape))
            Sx = np.stack([np.cov(elt) for elt in values.transpose(1, 2, 0)], axis=0)
            det = np.log(np.linalg.det(Sx))
            HxHy = np.add.outer(det, det)
            Hxy = np.zeros_like(HxHy)
            np.fill_diagonal(HxHy, 0)
            iterator = list(cwr(range(n_features), 2))

            for ij in tqdm(iterator):
                i, j = ij
                covs = np.cov(np.concatenate([values[:, i, :], values[:, j, :]], axis=-1).T)
                _det = np.linalg.det(covs)
                Hxy[[i, j], [j, i]] = _det

            np.fill_diagonal(Hxy, 1)
            Hxy = np.log(Hxy)
            mutual_information = 1 / 2 * (HxHy - Hxy)
            if correction:
                mutual_information -= np.min(mutual_information)
        return mutual_information


def _digamma_n(x, e, offset=1):
    """
    Computes the average digamma of the number of points between x at a distance e. Used in the k-nn estimator for correlation.

    See Also
    --------
    ``[**]`` function adapted from https://github.com/agheeraert/pmdlearn/blob/d554d4f02f46c0f5df4428a2a04549397cf13244/src/pmdlearn/features.py
    """

    tree = KDTree(x, metric='chebyshev')
    # We need to cheat using e-1e-10 so that we don't count kth neighbor
    # and finding in which dimension(s) the kth neighbor reaches e is too
    # expensive
    num_points = tree.query_radius(x, e - 1e-10, count_only=True)

    # Removing a count for self only if it is not null
    num_points = num_points[num_points.nonzero()] - 1

    # There's an extra count in num_points so we take it into account
    num_points += offset
    return digamma(num_points)


def _mutual_information_knn_estimator(values, features_dimension=1, k=5, estimate=1,
                                      correction=True, subset=None):
    """
    Knn estimator to evaluate mutual information. Is intrinsically non-linear but has long computation time.
    Implemetation of equation (8) and (9) in Kraskov et al., PHYSICAL REVIEW E 69, 066138 (2004).
    ``[**]``

    Parameters
    ----------
    values: np.array,
        values should be in format (n_samples, n_features, feature_dimension)
    features_dimension: int
        features dimensions (i.e. 3 for xyz coordinates)
    k: int
        nearest neighbours
    estimate: int (1 or 2)
        - offset=1, constant=0
        - offset=0, constant=1/k
    correction
        whether to apply correction (subtraction of min value of mutual information)
    subset: np.array (or None)
        indices of features for which to calculate MI, if None, all features are kept

    Returns
    -------
    mutual_information: np.ndarray, shape (nfeatures, nfeatures)
        mutual information matrix

    See Also
    --------
    ``[**]`` function adapted from https://github.com/agheeraert/pmdlearn/blob/d554d4f02f46c0f5df4428a2a04549397cf13244/src/pmdlearn/features.py
    """

    # The precise knn estimator changes one offset in the digamma function
    # and adds a constant in some cases.

    if estimate == 1:
        print('@>: estimate == 1')
        offset = 1
        costant = 0
    elif estimate == 2:
        print('@>: estimate == 2')
        offset = 0
        costant = -1 / k

    elif not (estimate == 1 or estimate == 2):
        print('@>: defaulting to estimate == 1')
        offset = 1
        costant = 0

    n_features = values.shape[1]

    # Initializing mutual information
    mutual_information = np.ones((n_features, n_features))
    mutual_information *= digamma(k) + digamma(values.shape[0]) + costant

    if subset is None:
        iterator = list(cwr(range(n_features), 2))
    else:
        iterator = subset

    if features_dimension == 1:
        values = np.asarray(values)[:, :, np.newaxis]

    for ij in tqdm(iterator):
        i, j = ij
        x = values[:, i, :]
        y = values[:, j, :]

        points = np.concatenate([x, y], axis=1)
        # Building KDTree with max norm metric
        tree = KDTree(points, metric='chebyshev')
        # Adding an offset to not count self as nearest neighbor
        e = tree.query(points, k=k + 1)[0][:, k]
        digamma_nxny = (_digamma_n(x, e, offset=offset) +
                        _digamma_n(y, e, offset=offset))
        digamma_nxny = digamma_nxny[np.isfinite(digamma_nxny)]
        mutual_information[[i, j], [j, i]] -= np.average(digamma_nxny)

    if correction:
        mutual_information -= np.min(mutual_information)
    return mutual_information


def compute_generalized_correlation_coefficients(values, features_dimension=1, solver='gaussian', correction=True, subset=None):
    """
    Rescale mutual information based correlation (`gcc`) between 0 and 1

    Parameters
    ----------
    values: np.ndarray, shape (nsamples, nfeatures, features_dimension)
         time-series values
    features_dimension: int,
        dimensionality of features array
    solver: string,
        which solver to use, default 'gaussian'
    correction: bool,
         whether to apply correction (subtraction of min value of mutual information)
    subset: np.array or None,
        indices of features for which to calculate `gcc`, if None, all features are kept

    Returns
    -------
    gcc: np.ndarray,
        generalized correlation coefficient matrix
    """

    MI = None
    if solver == 'gaussian':
        MI = _mutual_information_gaussian_estimator(values, features_dimension=features_dimension, correction=correction)

    elif 'knn' in solver:
        # Knn estimator is a string with two arguments encoded: knn_arg1_arg2,
        # arg1 is an integer representing the number of nn
        # arg2 is an integer that encodes the estimator used (1 or 2)
        k = int(solver[4])
        if len(solver) <= 6:
            estimate = 1
        else:
            estimate = int(solver[6])
        MI = _mutual_information_knn_estimator(values=values, features_dimension=features_dimension, k=k, estimate=estimate, correction=correction, subset=subset)

    gcc = np.sqrt(1 - np.exp(-2 * MI / features_dimension))
    np.fill_diagonal(gcc, 1)
    return gcc


@numba.njit
def compute_DCC_matrix(values, values_avg, features_dimension):
    """
    Compute dynamical cross-correlation matrix (upper triangle)
    ``[***]``

    Parameters
    ----------
    values: np.ndarray, shape (nsamples, nfeature, features_dimension)
        time-series values
    values_avg: np.ndarray, shape (nfeature, features_dimension)
        mean time-series values (values averaged over time)
    features_dimension: int,
        dimensionality of features array

    Returns
    -------
    cross_correlation: np.ndarray, shape (nfeatures, nfeatures)
        dynamical cross correlation matrix


    See Also
    --------
    ``[***]`` function adapted from https://github.com/tekpinar/correlationplus/blob/master/correlationplus/calculate.py
    """

    nsamples  = values.shape[0]
    nfeatures = int(values.shape[1]/features_dimension)

    cross_correlation = np.zeros((nfeatures, nfeatures), dtype=np.double)
    for k in range(0, len(values)):
        disp_ = np.subtract(values[k], values_avg)
        for i in range(0, nfeatures):
            idx_xyz_i = features_dimension * i
            for j in range(i, nfeatures):
                idx_xyz_j = features_dimension * j
                if features_dimension == 3:
                    cross_correlation[i][j] += (disp_[idx_xyz_i] * disp_[idx_xyz_j] +
                                                disp_[idx_xyz_i + 1] * disp_[idx_xyz_j + 1] +
                                                disp_[idx_xyz_i + 2] * disp_[idx_xyz_j + 2])
                elif features_dimension == 4:
                    cross_correlation[i][j] += (disp_[idx_xyz_i] * disp_[idx_xyz_j] +
                                                disp_[idx_xyz_i + 1] * disp_[idx_xyz_j + 1] +
                                                disp_[idx_xyz_i + 2] * disp_[idx_xyz_j + 2] +
                                                disp_[idx_xyz_i + 3] * disp_[idx_xyz_j + 3])
                elif features_dimension == 2:
                    cross_correlation[i][j] += (disp_[idx_xyz_i] * disp_[idx_xyz_j] +
                                                disp_[idx_xyz_i + 1] * disp_[idx_xyz_j + 1])
                elif features_dimension == 1:
                    cross_correlation[i][j] += (disp_[idx_xyz_i] * disp_[idx_xyz_j] )
                else:
                    print('cross_correlation cannot be computed for the specified features dimension')
                    break

    return cross_correlation

def compute_DCC(values_, features_dimension, normalized=True):
    """
    Compute normalized dynamical pairwise cross-correlation coefficients for each given replica

    Parameters
    ----------
    values_: np.ndarray, shape (nframes, natoms*features_dimensions)
        coordinates array
    features_dimension: int,
        dimension of features array
    normalized: bool,
        whether to normalize cross-correlation matrix; default is True

    Returns
    -------
    cross_correlation: np.ndarray, shape (natoms, natoms)
        cross-correlation matrix
    """

    # number of frames
    num_frames = values_.shape[0]
    nfeatures = int(values_.shape[1] / features_dimension)

    # exclude last entry of coordinates array - it contains the mean.
    values_average = np.mean(values_, axis=0)

    # compute DCC matrix
    cross_correlation = compute_DCC_matrix(np.asarray(values_, np.double),
                                            np.asarray(values_average, np.double), features_dimension)

    # Do the averaging
    cross_correlation = cross_correlation / float(num_frames)

    if normalized:
        # normalize
        norm_cross_correlation = np.zeros((nfeatures, nfeatures), dtype=np.double)
        for i in range(0, nfeatures):
            for j in range(i, nfeatures):
                norm_cross_correlation[i][j] = cross_correlation[i][j] / (
                            (cross_correlation[i][i] * cross_correlation[j][j]) ** 0.5)
                norm_cross_correlation[j][i] = norm_cross_correlation[i][j]
        cross_correlation = norm_cross_correlation
        return cross_correlation

    else:
        for i in range(0, nfeatures):
            for j in range(i + 1, nfeatures):
                cross_correlation[j][i] = cross_correlation[i][j]
        return cross_correlation



def corr(mat):
    """
    Compute pearson correlation

    Parameters
    ----------
    mat: np.ndarray,
        is an input matrix of shape (nsamples, nfeatures)

    Returns
    -------
    corrmat: np.ndarray, shape (nfeatures, nfeatures)
        pearson correlation matrix
    """
    corrmat = np.corrcoef(mat, rowvar=False)
    np.fill_diagonal(corrmat, 1)
    return corrmat


def prune_adjacency(mat, distmat, loc_factor=5.0, greater=False, lower=False):
    """
    Zero out adjacency matrics value according to the locality factor. By defalult entries corresponding to atom pairs at a distance equal or greater than the locality_factor (loc_factor) are set to zero.

    Parameters
    ----------
    mat: np.ndarray
        adjacency matrix
    distmat: np.ndarray
        distance matrix
    loc_factor: float x
        locality_factor, defines the distance threshold to use for pruning
    greater: bool
        whether to prune residues pairs at distances GREATER than the locality factor, default is False
    lower: bool
        whether to prune residue pairs at distances LOWER than the locality factor, default is False

    Returns
    --------
    mat: np.ndarray,
        pruned ajacency matrix
    """

    if greater:
        mat[distmat > loc_factor] = 0.
    elif lower:
        mat[distmat < loc_factor] = 0.
    else:
        print('@>: prune values greater than %d by default' % loc_factor)
        mat[distmat > loc_factor] = 0.

    return mat


def filter_adjacency(mat, distmat, loc_factor):
    """
    Filter the adjacency matrix using exponential to dump long-range contacts (and hence focus on short-range ones)

    Parameters
    ----------
    mat: np.ndarray,
        matrix
    distmat: np.ndarray,
        matrix based on which to prune mat
    loc_factor: float,
        locality factor (in Amstrongs), suggested 5 Å

    Returns
    -------
    adj_matx: np.ndarray,
        filtered matrix
    """

    distmat = distmat
    adj_matx = np.zeros(mat.shape)
    for i in range(0, len(mat)):
        adj_matx[i, i] = 0.0
        distmat[i, i] = 0.0
        for j in range(0, len(mat)):
            adj_matx[i, j] = mat[i, j] * np.exp(-1.0 * distmat[i, j] / loc_factor)
            adj_matx[j, i] = adj_matx[i, j]
    return adj_matx



def filter_adjacency_inv(mat, distmat, loc_factor):
    """
    Filter the adjacency matrix using exponential to dump short-range contacts
    (and hence focus on long-range ones)

    Parameters
    ----------
    mat: np.ndarray,
        matrix
    distmat: np.ndarray,
        matrix based on which to prune mat
    loc_factor: float,
        locality factor (in Amstrongs), suggested 5 Å

    Returns
    -------
    adj_matx: np.ndarray,
        filtered matrix
    """

    distmat = distmat
    adj_matx = np.zeros(mat.shape)
    for i in range(0, len(mat)):
        adj_matx[i, i] = 0.0
        distmat[i, i] = 0.0
        for j in range(0, len(mat)):
            adj_matx[i, j] = mat[i, j] * np.exp(-1.0 * loc_factor/ distmat[i, j])
            adj_matx[j, i] = adj_matx[i, j]
    return adj_matx


def reduce_trajectory(universe, segIDs):
    """
    It can be useful to work on a reduced trajectory without hydrogens, especially when visualizing stuff.
    This function can be used to generate a reduced MDAnalysis universe.

    Parameters
    ----------
    universe: MDAnalysis.core.Universe object
    segIDs: str,
        segIDs of groups of atoms to keep

    Returns
    -------
    reduced: MDAnalysis.core.Universe object
        reduced universe
    """

    from MDAnalysis.analysis.base import AnalysisFromFunction as mdaAFF
    from MDAnalysis.coordinates.memory import MemoryReader as mdaMemRead


    print("@>: the initial universe had {} atoms.".format(len(universe.atoms)))

    # initial selection
    sel_str = "segid " + " ".join(segIDs)
    group_sel = universe.select_atoms(sel_str)
    group_sel_noh = group_sel.select_atoms("not type H*")

    # merging a selection from the universe returns a new (and smaller) universe
    reduced = mda.core.universe.Merge(group_sel_noh)

    print("@>: the final universe has {} atoms.".format(len(reduced.atoms)))

    # load the new universe to memory, with coordinates from the selected residues
    print("@>: loading universe to memory...")

    reduced.load_new(mdaAFF(lambda ag: ag.positions.copy(),
                            group_sel_noh).run().results, format=mdaMemRead)
    return reduced


def linfit(x, y):
    """
    Return R^2 where x and y are array-like
    """

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return slope, intercept, r_value**2


def create_graph(adj_matx):
    """
    Create nx.Graph instance from a np.ndarray adjacency_matrix
    """
    G = nx.Graph(adj_matx)
    G_ = G.copy().to_undirected()
    G_.remove_edges_from(nx.selfloop_edges(G_))
    G = G_.copy()
    return G


def vec_query(arr, my_dict):
    """
    Convert dictionary to array
    """
    return np.vectorize(my_dict.__getitem__)(arr)



def compute_eigenvector_centrality(mat, loc_factor=None, distmat=None, weight='weight'):
    """
    Compute eigenvector centrality based on a given correlation matrix and pairwise distance matrix

    Parameters
    ----------
    mat: np.ndarray,
        adiacency (correlation matrix) to diagonalize
    loc_factor: float,
        locality factor (filtering threshold)
    distmat: np.ndarray,
        distance matrix
    weight: str or None,
        if None treat adjaccency as binary, if 'weight' use values weights.

    Returns
    --------
    cdict: dict,
        centrality dictionary with nodes as keys and centrality coefficients as values
    cvec: np.ndarray,
        eigenvector centrality coefficients
    """

    if loc_factor is not None:
        adj_matx = filter_adjacency(mat, loc_factor=loc_factor, distmat=distmat)
    else:
        adj_matx = mat

    G = create_graph(adj_matx)
    # calculate all node centrality values in the system.
    cdict = nx.eigenvector_centrality(G, max_iter=10000, weight=weight)
    all_nodes = np.array(list(cdict.keys()))
    cvec = vec_query(all_nodes, cdict)

    return cdict, cvec


def sorted_eig(A):
    """
    Sort eigenvalues from larges to smallest and reorder eigenvectors accordingly

    Parameters
    -----------
    A: array, shape (nfeatures, nfeatures)
                array of eigenvalues

    Returns
    -------
     eigenValues: np.ndarray, shape (nfeatures)

     eigenVectors: np.ndarray, shape (nfeatures,nfeatures)
    """

    eigenValues, eigenVectors = scipy.linalg.eigh(A)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]

    return eigenValues, eigenVectors


def to_pickle(dataframe, output):
    """
    Dump dataframe to file using pickle

    Parameters
    -----------
    dataframe: pd.DataFrame,
        dataframe to pickle
    output: str,
        output name with path
    """

    pickle.dump(dataframe, open(output, 'wb'))


def get_centrality_df(input_class, cent_don, cent_acc, cent, selection):
    """
    Create dataframe containing eigenvector centrality from KS energy and from alpha carbon displacements

    Parameters
    ----------
    input_class: object,

    cent_don: np.ndarray,

    cent_acc: np.ndarray,

    cent: np.ndarray,

    selection: str,

    Returns
    -------
    temp_df:  pd.DataFrame()
        data frame containing electrostatic-centrality and CA-centrality
    """

    temp_df = pd.DataFrame()
    temp_df['Res. Name'] = input_class.mda_u.select_atoms(selection).residues.resnames
    temp_df['Res. Num.'] = input_class.mda_u.select_atoms(selection).residues.resnums
    temp_df['Acceptor Centrality'] = cent_acc
    temp_df['Donor Centrality'] = cent_don
    temp_df['Electrostatic Centrality'] = cent_don + cent_acc
    temp_df['CA Centrality'] = cent
    return temp_df




