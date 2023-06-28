"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: fmaschietto, bcallen95"""

from   mdigest.core.imports import *
import mdigest.core.toolkit as tk
import mdigest.core.savedata as sd
import mdigest.core.auxiliary as aux

from operator                 import itemgetter as ig
from sklearn.preprocessing    import StandardScaler
from MDAnalysis.lib.distances import capped_distance

class DynCorr:
    """General purpose class handling computation of different correlation metrics from atomic displacements sampled over MD trajectories."""

    def __init__(self, MDSIM):
        """
        Description
        -----------
        General purpose class handling computation of different correlation metrics from atomic displacements sampled over MD trajectories.

        Parameters
        ----------
        MDSIM: class object
        DynCorr inherits general attributes from MDSIM


        Methods
        -------


        Attributes
        ----------

        self.coordinates_allreplicas: dict, with replica index ``rep_n`` as key and values of shape (nsamples, nfeatures * features_dimension)
            coordinates array.

        self.displacements_allreplicas: dict, with replica index ``rep_n`` as key and values of shape shape (nsamples, nfeatures * features_dimension)
            array containing the displacement of each residue from the average position (the mean x, y, z coordinates
            over all the selected timesteps)

        self.distances_allreplicas: dict, with replica index ``rep_n`` as key and values of shape (nfeatures, nfeatures)
            node to node pairwise distances computed from average positions for each given trajectory replica

        self.disp_from_mean_allreplicas: dict, with replica index ``rep_n`` as key and values of shape (nsamples, nfeatures)
            array containing the displacement of each atom from the average position computed over all the selected timesteps

        self.covar_disp_allreplicas: dict, with replica index ``rep_n`` as key and values of shape (nfeatures, nfeatures)
            covariance matrix of atomic displacements for each given trajectory replica

        self.gcc_allreplicas: nested dict, with replica index ``rep_n`` as key and inner dict with key ``gcc_mi`` or ``gcc_lmi`` and values of shape (nfeatures, nfeatures)
            linearized Mutual Information based generalized correlation coefficient matrix[1]. Each entry represents the pairwise linearized generalized correlation coefficient between each pair of nodes,
            for each given trajectory replica. ``gcc_lmi`` linearized mutual information based generalized correlation using gaussian estimator; ``gcc_mi`` mutual information based generalized correlation computed using nonlinear estimator.

        self.dcc_allreplicas: dict, with replica index ``rep_n`` as key and values of shape (nfeatures, nfeatures)
            normalized dynamical cross-correlations matrix

        self.pcc_allreplicas: dict, with replica index ``rep_n`` as key and values of shape (nfeatures, nfeatures)
            Pearson's product-moment correlation coefficients.

        self.exclusion_matrix_allreplicas: dict, with replica index ``rep_n`` as key and values of shape (nfeatures, nfeatures)
            which nodes to consider for calculation of communities

        self.eigenvector_centrality_allreplicas: dict, with replica index ``rep_n`` as key and values of shape (nfeatures)
            eigenvector centrality arrays for each given trajectory replica


        References
        ----------
        [1] Lange, O.; Grubmueller, H.: Generalized correlation for biomolecular dynamics.
        Proteins: Structure, Function and Bioinformatics 62 (4), pp. 1053 - 1061 (2006)


        Examples
        --------
        """
        self.mds_data                = MDSIM.mds_data
        self.mda_u                   = MDSIM.mda_u
        self.atom_group_selstr       = MDSIM.atom_group_selstr
        self.system_selstr           = MDSIM.system_selstr
        self.atom_group_selection    = MDSIM.atom_group_selection
        self.nodes_idx_array         = MDSIM.nodes_idx_array
        self.nodes_to_res_dictionary = MDSIM.nodes_to_res_dictionary

        # ---------------------------------------------#

        self.coordinates_allreplicas            = {}
        self.displacements_allreplicas          = {}
        self.distances_allreplicas              = {}
        self.disp_from_mean_allreplicas         = {}
        self.covar_disp_allreplicas             = {}
        self.gcc_allreplicas                    = {}
        self.dcc_allreplicas                    = {}
        self.pcc_allreplicas                    = {}
        self.exclusion_matrix_allreplicas       = {}
        self.eigenvector_centrality_allreplicas = {}

        # ---------------------------------------------#

        self.natoms              = MDSIM.natoms
        self.total_nframes       = MDSIM.total_nframes
        self.nframes_per_replica = MDSIM.nframes_per_replica
        self.num_replicas        = MDSIM.num_replicas
        self.nresidues           = MDSIM.nresidues
        self.nnodes              = MDSIM.nnodes # nfeatures
        self.initial             = MDSIM.initial
        self.final               = MDSIM.final
        self.step                = MDSIM.step
        self.window_span         = MDSIM.window_span # nsamples

    def save_class(self, file_name_root='./output/cache/'):
        """
        can be used to dump all correlation analyses (produced upon calling the correlation, kscorrelation,
        and dcorrelation modules) or the community analysis.

        Parameters
        ----------
        file_name_root: str
            filename rootname
        """

        self.mds_data = sd.MDSdata()
        self.mds_data.atom_group_selstr                  = self.atom_group_selstr
        self.mds_data.system_selstr                      = self.system_selstr
        self.mds_data.nodes_idx_array                    = self.atom_group_selection.ix_array
        self.mds_data.nodes_to_res_dictionary            = self.nodes_to_res_dictionary # useful for later plotting (match nodes id to resid)
        self.mds_data.nresidues                          = self.nresidues
        self.mds_data.nnodes                             = self.nnodes
        self.mds_data.num_replicas                       = self.num_replicas
        self.mds_data.disp_from_mean_allreplicas         = self.disp_from_mean_allreplicas
        self.mds_data.covar_disp_allreplicas             = self.covar_disp_allreplicas
        self.mds_data.coordinates_allreplicas            = self.coordinates_allreplicas
        self.mds_data.displacements_allreplicas          = self.displacements_allreplicas
        self.mds_data.distances_allreplicas              = self.distances_allreplicas
        self.mds_data.gcc_allreplicas                    = self.gcc_allreplicas
        self.mds_data.dcc_allreplicas                    = self.dcc_allreplicas
        self.mds_data.pcc_allreplicas                    = self.pcc_allreplicas
        self.mds_data.exclusion_matrix_allreplicas       = self.exclusion_matrix_allreplicas
        self.mds_data.eigenvector_centrality_allreplicas = self.eigenvector_centrality_allreplicas
        self.mds_data.save_to_file(file_name_root)


    def edge_exclusion(self, spatial_cutoff=4.5, contact_cutoff=.75, save_name='none'):
        # TODO priority 1 MOVE THIS TO COMMUNITIES MODULE
        # TODO change nresidues with nnodes.
        # Warning: this function was only tested with the option of number of nodes in the calculation chosen equal to
        # nresidues

        """
        Compute the so-called `edge exclusion matrix`. When analyzing correlations it can be useful to analyze only correlation corresponding to residue pairs that
        are proximal (below a certain distance threshold) for a certain percentage of the trajectory frames. This function takes care of computing this using `capped_distance` from MDAnalysis.

        Parameters
        ----------
        spatial_cutoff: float,
            distance threshold to define atoms in contact (in Amstrong)
        contact_cutoff: float,
            contact persistency above which to consider pair expressed in percentage of frames
        save_name:
            output filename
        """

        def _multiplicity_pairs(pair_array, threshold):
            x = pair_array.transpose()[0]
            y = pair_array.transpose()[1]
            m = np.max(pair_array) * 10.0
            h = x * m + y
            unique, count = np.unique(h, return_counts=True)
            good_pairs_h = unique[count >= threshold]
            x_g = (np.floor(good_pairs_h / float(m))).astype(int)
            y_g = (good_pairs_h % m).astype(int)
            good_pairs = np.transpose(np.vstack([x_g, y_g]))
            return good_pairs

        def _unique_pairs(pair_array):
            x = pair_array.transpose()[0]
            y = pair_array.transpose()[1]
            m = 1000000
            h = x * m + y
            unique, count = np.unique(h, return_counts=True)
            x_g = (np.floor(unique / float(m))).astype(int)
            y_g = (unique % m).astype(int)
            return np.transpose(np.vstack([x_g, y_g]))

        def _compute_exclusion_matrix(universe, sys_sele_str, nres, i_frame, f_frame, n_space, spatialcutoff,
                                      contactcutoff, savename='None'):
            """
            Computation of exclusion matrix. The function is fed with instances from MDSIM class / arguments at provided at call
            TODO cleanup, convert function into class method

            Parameters
            ----------
            universe: object,
                MDA universe
            sys_sele_str:    str,
                defining the subset of atoms used in the computation of the exclusion matrix
            nres:            int,
                number of residues (nodes)
            i_frame:         int,
                initial frame of the trajectory used for exclusion matrix calculation
            f_frame:         int,
                final frame of the trajectory used for exclusion matrix calculation
            n_space:         int,
                trajectory step used for exclusion matrix calculation
            spatialcutoff:   float,
                distance threshold defining whether two atoms (nodes) are in contact
            contactcutoff:   float,
                contact persistency threshold defining whether two atoms (nodes) are in contact
            savename:        str,
                output filename

            Returns
            -------
            excl_mat: np.ndarray
                exclusion matrix
            """

            if savename != 'None':
                print('@>: saving exclusion matrixt to ', savename)
                m = tk.retrieve(savename + ".eemat")

                if m is not None:
                    return m

            residues = universe.select_atoms(sys_sele_str, updating=True).residues

            natoms = len(universe.select_atoms(sys_sele_str, updating=True).residues.atoms)

            print('@>: number of selected atoms: ',
                  len(universe.select_atoms(sys_sele_str, updating=True).residues.atoms.ids))

            atom_to_res_dictionary = dict(zip(universe.select_atoms(sys_sele_str).atoms.ids,
                                              universe.select_atoms(sys_sele_str).residues.atoms.resindices))

            residue_name_to_index = {}
            for indx in range(len(residues)):
                residue_name_to_index[residues[indx].resindex] = indx

            inclusion_matrix = np.zeros((nres, nres))
            mega_list = []
            noth_ids = residues.atoms.select_atoms("not type H*").ids

            print("---------------------------------------------------------------")
            print("@>: computing exclusion matrix...")
            print("@>: first:  %d, last: %d, step: %d" % (i_frame, f_frame, n_space))
            print("---------------------------------------------------------------")

            trajectory = universe.trajectory[i_frame:f_frame:n_space]
            trajlen = len(trajectory)
            for ts in trajectory:
                not_h = residues.atoms.select_atoms("not type H*", updating=True)

                pairs = capped_distance(not_h.positions, not_h.positions, spatialcutoff, return_distances=False)

                second_atoms = np.transpose(pairs)[1]
                first_atoms = np.transpose(pairs)[0]

                second_atoms = noth_ids[second_atoms]
                first_atoms = noth_ids[first_atoms]

                a2r_selection1 = np.asarray(ig(*first_atoms)(atom_to_res_dictionary))
                a2r_selection2 = np.asarray(ig(*second_atoms)(atom_to_res_dictionary))

                res_pairs = np.transpose(np.vstack([a2r_selection1, a2r_selection2]))
                unique_res_pairs = _unique_pairs(res_pairs)
                mega_list.append(unique_res_pairs)

            mega_list = np.vstack(mega_list)
            good_atom_pairs = _multiplicity_pairs(mega_list, contactcutoff * trajlen)

            for pair in good_atom_pairs:
                res_1 = residue_name_to_index[pair[0]]
                res_2 = residue_name_to_index[pair[1]]
                inclusion_matrix[res_1, res_2] = 1
                inclusion_matrix[res_2, res_1] = 1

            excl_mat = 1 - inclusion_matrix
            for i in range(0, nres - 1):
                excl_mat[i, i + 1] = 1.0

            for i in range(1, nres):
                excl_mat[i, i - 1] = 1.0

            np.fill_diagonal(excl_mat, 1.0)

            if savename != 'None':
                out_name = save_name + '_%d' % win_idx + ".eemat"
                tk.dump(out_name, excl_mat)

            print("---------------------------------------------------------------")
            print("@>: done computing exclusion matrix.")
            print("@>: found ", (nres * nres) - np.sum(excl_mat), 'nonzero elements in the exclusion matrix')
            print("---------------------------------------------------------------")
            return excl_mat

        exclusion_matrix_allreplicas = {}

        for win_idx in tk.log_progress(range(self.num_replicas), every=1, size=self.num_replicas, name="Window"):
            #beg = int(self.final / self.num_replicas) * win_idx
            #end = int(self.final / self.num_replicas) * (win_idx + 1)

            offset =  (self.final - self.initial)// self.num_replicas
            if self.window_span != offset/self.step:
                print("@>: WARNING: the offset is not equal to the window span")

            beg = self.initial + offset * win_idx
            end = self.initial + offset * (win_idx + 1)

            print('@>: using frames %d to %d with step %s' % (beg, end, self.step))

            name = save_name + "_%d" % win_idx

            exclusion_matrix_allreplicas['rep_%d' %win_idx] = _compute_exclusion_matrix(self.mda_u, self.system_selstr,
                                                                              self.nresidues,
                                                                              beg, end, self.step, spatial_cutoff,
                                                                              contact_cutoff, name)

        self.exclusion_matrix_allreplicas = exclusion_matrix_allreplicas


    def parse_dynamics(self, scale=False, normalize=True, LMI='gaussian', MI='knn_5_1', DCC=False, PCC=False, COV_DISP=False,
                       CENTRALITY=True, VERBOSE=False, **kwargs):
        """
        Parse molecular dynamics trajectory and compute different correlation metrices

        Parameters
        ----------
        scale: bool,
            whether to remove mean from coordinates using StandardScaler

        normalize: bool,
            whether to normalize cross-correlation matrices

        LMI: str or None,
            - None to skip computation of LMI based correlation
            - 'gaussian' to compute LMI

        MI: str,
            -None to skip computation of MI based correlation
            -'knn_arg1_arg2' to compute MI, with k = arg1, and estimator= arg2, default is 'knn_5_1'

        DCC: bool,
            whether to compute dynamical cross-correlation matrix of atomic displacements. Default is False

        PCC: bool,
            whether to compute Pearson correlation matrix of atomic displacements. Default is False

        COV_DISP: bool,
            whether to compute the covariance of atomic displacements. Default is False

        CENTRALITY: bool,
            whether to compute centrality of atomic displacements. Default is True

        VERBOSE: bool,
            whether to set verbose printing

        """

        # Assign stride variable
        stride = self.step

        if self.nresidues != self.nnodes:
            nr = self.nnodes
            print('@>: sanity check warning: number of nodes included exceeds the number of residues')
        else:
            nr = self.nresidues
            print('@>: sanity check pass: number of residues is same as number of nodes')



        # Number or replicas
        num_replicas = self.num_replicas

        print("@>: using window length of %d simulation steps" % self.window_span)

        # For 3D atom position data
        feat_dimension = 3

        # Stores all data in a frame-by-dimension format.
        coordinates                = np.zeros((num_replicas, self.window_span, nr, feat_dimension),  dtype=float)
        coordinates_allreplicas    = np.zeros((num_replicas, self.window_span, nr * feat_dimension), dtype=float)
        displacements_allreplicas  = np.zeros((num_replicas, self.window_span, nr * feat_dimension))
        disp_from_mean_allreplicas = np.zeros((num_replicas, self.window_span, nr))
        covar_disp_allreplicas     = np.zeros((num_replicas, nr, nr))
        distances_allreplicas      = np.zeros((num_replicas, nr, nr))
        dcc_allreplicas            = np.zeros((num_replicas, nr, nr))
        pcc_allreplicas            = np.zeros((num_replicas, nr, nr))

        if VERBOSE:
            print('@> -- shape of coordinates matrix:',             coordinates.shape)
            print('@> -- shape of coordinates_allreplicas matrix:', coordinates_allreplicas.shape)
            print('@> -- shape of displacements_allreplicas matrix:', displacements_allreplicas.shape)
            print('@> -- shape of disp_from_mean_allreplicas matrix:', disp_from_mean_allreplicas.shape)
            print('@> -- shape of distances_allreplicas matrix:', distances_allreplicas.shape)
            print('@> -- shape of covar_disp_allreplicas', covar_disp_allreplicas.shape)
            print('@> -- shape of dcc_allreplicas matrix:', dcc_allreplicas.shape)
            print('@> -- shape of pcc_allreplicas matrix:', pcc_allreplicas.shape)


        # setup kwargs
        try:
            subset = kwargs['subset']
        except KeyError:
            subset = None

        solvers = []
        if MI  is not None:
            solvers.append(MI)
        if LMI is not None:
            solvers.append(LMI)


        for win_idx in tk.log_progress(range(num_replicas), every=1, size=num_replicas, name="Window"):

            #beg = int(self.final / self.num_replicas) * win_idx
            #end = int(self.final / self.num_replicas) * (win_idx + 1)
            offset =  (self.final - self.initial)// self.num_replicas
            if self.window_span != offset/self.step:
                print("@>: WARNING: the offset is not equal to the window span")

            beg = self.initial + offset * win_idx
            end = self.initial + offset * (win_idx + 1)


            print("@>: LMI/MI calculation ...")
            print("@>: begin frame: %d" % beg)
            print("@>: end   frame: %d" % end)
            print("@>: step:        %d" % self.step)

            counter = 0
            for frame in self.mda_u.trajectory[beg:end:stride]:
                coordinates[win_idx, counter, :, :] = self.atom_group_selection.positions
                counter += 1
            counter = 0

            if COV_DISP:
                print("@>: compute covariance of displacements...")

                disp_from_mean_allreplicas[win_idx] = aux._compute_square_displacements(coordinates[win_idx, :, :, :])

                if scale:
                    # nfeatures displacements from mean are not centered hence scaling will have an effect
                    tmp = disp_from_mean_allreplicas[win_idx].copy()
                    scaler = StandardScaler(with_std=False).fit(tmp)
                    disp_from_mean_allreplicas[win_idx] = scaler.transform(tmp)

                    covar_disp_allreplicas[win_idx] = np.cov(disp_from_mean_allreplicas[win_idx], rowvar=False, bias=True)

                else:
                    covar_disp_allreplicas[win_idx] = np.cov(disp_from_mean_allreplicas[win_idx], rowvar=False, bias=True)


            print("@>: reshaping coordinates...")

            coordinates_allreplicas[win_idx] = aux.coordinate_reshape(coordinates[win_idx, :, :, :])

            if scale:
                # scaling has no effect because 3*nfeatures displacements are already mean averaged
                c_ = coordinates[win_idx, :, :].copy()
                tmp = aux._compute_displacements(c_)
                scaler = StandardScaler(with_std=False).fit(tmp)
                displacements_allreplicas[win_idx] = scaler.transform(tmp)
            else:
                print("@>: computing the displacement vectors (X- <X>)...")
                displacements_allreplicas[win_idx] = aux._compute_displacements(coordinates[win_idx, :, :])

            MIdict = {}
            ECdict = {}
            if solvers is not []:
                for solver in solvers:
                    if solver == 'gaussian':
                        MIdict.update({'gcc_lmi': aux.compute_generalized_correlation_coefficients(displacements_allreplicas[win_idx].reshape((self.window_span, nr, feat_dimension)),
                                                                                         features_dimension=feat_dimension, solver=solver, correction=False)})
                        if CENTRALITY:
                            print("@>: computing eigenvector centrality from lmi matrix")
                            _, ec = aux.compute_eigenvector_centrality(MIdict['gcc_lmi'], weight='weight')
                            ECdict.update({'gcc_lmi': ec})




                    elif 'knn' in solver:
                        MIdict.update({'gcc_mi': aux.compute_generalized_correlation_coefficients(displacements_allreplicas[win_idx].reshape((self.window_span, nr, feat_dimension)),
                                                                         features_dimension=feat_dimension, solver=solver, correction=True, subset=subset)})
                        if CENTRALITY:
                            print("@>: computing eigenvector centrality from mi matrix")
                            _, ec = aux.compute_eigenvector_centrality(MIdict['gcc_mi'], weight='weight')
                            ECdict.update({'gcc_mi': ec})

            self.gcc_allreplicas['rep_%d' % win_idx] = MIdict
            self.eigenvector_centrality_allreplicas['rep_%d' % win_idx] = ECdict

            print("@>: computing and storing distances...")
            distances_allreplicas[win_idx] = aux.compute_distance_matrix(coordinates_allreplicas[win_idx, :, :], feat_dimension)

            if DCC:
                print("@>: computing normalized dynamical cross-correlation matrix")
                dcc_allreplicas[win_idx] = aux.compute_DCC(coordinates_allreplicas[win_idx], features_dimension=feat_dimension, normalized=normalize)

            if PCC:
                print("@>: computing pearson correlation matrix")
                pcc_allreplicas[win_idx] = np.corrcoef(disp_from_mean_allreplicas[win_idx], rowvar=False)

        # Store as class instances
        self.coordinates_allreplicas     = dict(zip(['rep_%d' % i for i  in range(self.num_replicas)], coordinates_allreplicas))
        self.displacements_allreplicas   = dict(zip(['rep_%d' % i for i  in range(self.num_replicas)], displacements_allreplicas))
        self.distances_allreplicas       = dict(zip(['rep_%d' % i for i  in range(self.num_replicas)], distances_allreplicas))
        self.dcc_allreplicas             = dict(zip(['rep_%d' % i for i  in range(self.num_replicas)], dcc_allreplicas))
        self.pcc_allreplicas             = dict(zip(['rep_%d' % i for i  in range(self.num_replicas)], pcc_allreplicas))
        self.disp_from_mean_allreplicas  = dict(zip(['rep_%d' % i for i  in range(self.num_replicas)], disp_from_mean_allreplicas))
        self.covar_disp_allreplicas      = dict(zip(['rep_%d' % i for i  in range(self.num_replicas)], covar_disp_allreplicas))

