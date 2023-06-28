"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: fmaschietto, bcallen95"""

from MDAnalysis.analysis.dihedrals import Ramachandran

import mdigest.core.auxiliary as aux
import mdigest.core.savedata as sd
import mdigest.core.toolkit as tk
from   mdigest.core.imports import *


class DihDynCorr:
    """Correlated motions of dihedrals"""
    # TODO CHANGE Nresidues to Natoms - to enable the calculation of correlation over different selections and more than one node per residue

    def __init__(self, MDSIM):
        """
         Description
         -----------
         General purpose class handling computation of different correlation metrics from dihedrals fluctuations sampled
         over MD trajectories. Each (selected) residue is described using the transformation --> [$sin(\phi)$, $cos(\phi)$, $sin(\psi)$, $cos(\psi)$]

         Parameters
         ----------
         MDSIM: class object
             DynCorr inherits general attributes from MDSIM

         Methods
         -------

         Attributes
         ----------
         self.nodes_dih_to_indices_dictionary: dict,
             nodes to indices dictionary; useful for later plotting (match dihedral nodes id to resid)
         self.dihedrals_allreplicas: dict, with replica index ``rep_n`` as key and values of shape (nsamples, nfeatures * features_dimension)
             being the projected dihedrals values array.
         self.disp_from_mean_dih_allreplicas: dict, with replica index ``rep_n`` as key and values of shape (nsamples, nfeatures)
             being the displacement of each atom from the average position computed over all the selected timesteps
         self.covar_dih_allreplicas: dict, with replica index ``rep_n`` as key and values of shape (nfeatures, nfeatures)
             being the covariance matrices of atomic displacements for each given trajectory replica
         self.dih_gcc_allreplicas: nested dict, with replica index ``rep_n`` as key and inner dict with key ``gcc_mi`` or ``gcc_lmi`` and values of shape (nfeatures, nfeatures)
             being the mutual information based generalized correlation coefficient matrix[1] for each replica. ``gcc_lmi`` linearized mutual information based generalized correlation using gaussian estimator; ``gcc_mi`` mutual information based generalized correlation computed using nonlinear estimator.
         self.dih_dcc_allreplicas: dict, with replica index ``rep_n`` as key and values of shape (nfeatures, nfeatures)
             being the normalized dynamical cross-correlations matrices for each replica
         self.dih_pcc_allreplicas: dict, with replica index ``rep_n`` as key and values of shape (nfeatures, nfeatures)
             being the Pearson's product-moment correlation coefficients for each replica.
         self.ramachandran: object,
             mda.Ramachandran output
         self.dih_values: np.ndarray,
             dihedrals array
         self.dih_labels:  list,
             dihedrals labels
         self.dih_indices: np.ndarray,
             array of indices corresponding to the selected dihedrals

         References
         ----------

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
        self.natoms              = MDSIM.natoms
        self.total_nframes       = MDSIM.total_nframes
        self.nframes_per_replica = MDSIM.nframes_per_replica
        self.num_replicas        = MDSIM.num_replicas
        self.nresidues           = MDSIM.nresidues
        self.initial             = MDSIM.initial
        self.final               = MDSIM.final
        self.step                = MDSIM.step
        self.window_span         = MDSIM.window_span

        # ---------------------------------------------#
        self.ramachandran      = None
        self.dih_values        = None
        self.dih_labels        = None
        self.dih_indices       = None

        self.nodes_dih_to_indices_dictionary  = {}  # useful for later plotting (match dihedral nodes id to resid)
        self.dihedrals_allreplicas            = {}
        self.disp_from_mean_dih_allreplicas   = {}
        self.covar_dih_allreplicas            = {}
        self.dih_gcc_allreplicas              = {}
        self.dih_dcc_allreplicas              = {}
        self.dih_pcc_allreplicas              = {}
        self.dih_eigen_centrality_allreplicas = {}

    def save_class(self, file_name_root='./output/cache/'):
        """
        Save DihDynCorr class instances to file

        file_name_root: srt
            path where to save class
        """

        self.mds_data = sd.MDSdata()
        self.mds_data.nodes_dih_to_indices_dictionary = self.nodes_dih_to_indices_dictionary
        self.mds_data.nodes_to_res_dictionary = self.nodes_to_res_dictionary
        self.mds_data.atom_group_selstr = self.atom_group_selstr
        self.mds_data.system_selstr = self.system_selstr
        self.mds_data.dih_labels = self.dih_labels
        self.mds_data.dih_indices = self.dih_indices
        self.mds_data.dihedrals_allreplicas = self.dihedrals_allreplicas
        self.mds_data.disp_from_mean_dih_allreplicas = self.disp_from_mean_dih_allreplicas
        self.mds_data.covar_dih_allreplicas = self.covar_dih_allreplicas
        self.mds_data.dih_gcc_allreplicas = self.dih_gcc_allreplicas
        self.mds_data.dih_dcc_allreplicas = self.dih_dcc_allreplicas
        self.mds_data.dih_pcc_allreplicas = self.dih_pcc_allreplicas
        self.mds_data.dih_eigen_centrality_allreplicas = self.dih_eigen_centrality_allreplicas
        self.mds_data.save_to_file(file_name_root)


    def parse_dih_dynamics(self, mean_center=False, LMI='gaussian', MI='knn_5_1', DCC=False, PCC=False, COV_DISP=True, CENTRALITY=True, **kwargs):
        """
        General purpose class handling computation of different correlation metrics from $\phi$, \$psi$ backbone dihedrals fluctuations sampled over MD trajectories.
        Diedrals are transformed using $\phi$ --> {$sin(\phi)$, $cos(\phi)$} and $\psi$ --> {$sin(\psi)$, $cos(\psi)$} such that each residue
        (temimal residues excluded) is described by an array of four entries [$sin(\phi)$, $cos(\phi)$, $sin(\psi)$, $cos(\psi)$].

        Parameters
        ----------
        mean_center: bool
            wheter to subtract mean
        LMI: str or None; default 'gaussian'
            - 'gaussian' for using gaussian estimator
            - None: skip computation of linearized mutual information based correlation

        MI: str or None, default 'knn_5_1'
            composite argument where knn specifiess use of k-nearest neighbor algorithm,
            5 specifies number of nearest neighbours, 1 specifies estimate to use (options are 1 or 2)
        DCC: bool,
            whether to compute dynamical cross correltaion
        PCC: bool,
            whether to compute Pearson's cross correlation
        COV_DISP: bool,
            whether to compute covariance of dihedrals displacements
        CENTRALITY: bool,
            whether to compute eigen centrality of dihedrals fluctuations. Default is True
        kwargs:
            - normalized: bool
                whether to normalize DCC matrix
            - subset: list,
                list of indices specifying the nodes for which to compute MI
            - center: str or None
                How to compute the covariance matrix; possible values are 'mean' or 'square_disp'
            - reference_frame: bool,
                Standardize dihedrals by centering distribution to a frame of reference
            - median_center: bool,
                Standardize dihedrals by centering distribution to the median

        Returns
        -------

        """

        # setup kwargs
        if DCC:
            try:
                normalized = kwargs['normalized']
            except KeyError:
                print('@>: DCC normalization is set to True. Use normalize=False to avoid normalization.')
                normalized = True

        if COV_DISP:
            try:
                center = kwargs['center']  # None, mean or square_disp
            except KeyError:
                print('@>: Use center to select different ways to compute the covariance'
                      'possible options are None, mean or square_disp')
                center = 'square_disp'
                print('@>: center  = ', center)

        # setup kwargs
        try:
            subset = kwargs['subset']
        except KeyError:
            subset = None

        try:
            reference_frame = kwargs['reference_frame']
        except KeyError:
            reference_frame = 0

        try:
            median_center = kwargs['median_center']
        except KeyError:
            median_center = False

        try:
            reference_to_frame = kwargs['']
        except:
            reference_to_frame = False

        solvers = []
        if MI is not None:
            solvers.append(MI)
        if LMI is not None:
            solvers.append(LMI)

        num_replicas = self.num_replicas

        for win_idx in tk.log_progress(range(self.num_replicas), 1, size=num_replicas, name="Window"):

            #beg = int(self.final / self.num_replicas) * win_idx
            #end = int(self.final / self.num_replicas) * (win_idx + 1)
            offset =  (self.final - self.initial)// self.num_replicas
            if self.window_span != offset/self.step:
                print("@>: WARNING: the offset is not equal to the window span")

            beg = self.initial + offset * win_idx
            end = self.initial + offset * (win_idx + 1)

            stride = self.step
            print("@>: start, end frames:", beg, end)

            print("@>: Dihedrals calculation ...")
            self.ramachandran = Ramachandran(self.mda_u.universe.select_atoms(self.system_selstr)).run()
            r = self.ramachandran


            if reference_to_frame:
                # reference to selected frame
                dih_reference = r.angles[reference_frame, :]
                cos = np.cos(np.radians(r.angles - dih_reference))
                sin = np.sin(np.radians(r.angles - dih_reference))
                self.dih_values = np.concatenate([cos, sin], axis=2)
                mean_center = False

            if median_center:
                # subtract median
                median = np.median(r.angles, axis=0)
                cos = np.cos(np.radians(r.angles - median))
                sin = np.sin(np.radians(r.angles - median))
                self.dih_values = np.concatenate([cos, sin], axis=2)
                mean_center = False
                reference_to_frame = False

            if mean_center:
                # if no referencing to initial or other frame, and mean_center is True center with respect to average
                mean_angles = np.mean(r.angles.mean(axis=0)[beg:end:stride, :], axis=0)

                cos = np.cos(np.radians(r.angles - mean_angles))
                sin = np.sin(np.radians(r.angles - mean_angles))
                self.dih_values = np.concatenate([cos, sin], axis=2)

            else:
                # plain dihedrals
                cos = np.cos(np.radians(r.results.angles))
                sin = np.sin(np.radians(r.results.angles))
                self.dih_values = np.concatenate([cos, sin], axis=2)

            self.dih_labels = ['cos_phi', 'cos_psi', 'sin_phi', 'sin_psi']
            self.dih_indices = np.stack([r.ag1.residues.resindices, r.ag4.residues.resindices], axis=-1)
            features_dim = len(self.dih_labels)

            # use first column of dih_indices for mapping
            self.nodes_dih_to_indices_dictionary = dict(
                zip(self.dih_indices[:, 0], [res.phi_selection().atoms.ids for res
                                             in self.mda_u.residues if
                                             res.phi_selection() is not None]))

            print("@>: LMI calculation ...")
            print("@>: begin frame: %d" % beg)
            print("@>: end   frame: %d" % end)
            print("@>: step:        %d" % self.step)

            values = self.dih_values[beg:end:stride, :, :]
            print("@>: dih matrix shape: {}".format(values.shape))

            self.dihedrals_allreplicas['rep_%d' % win_idx] = values

            MIdict = {}
            ECdict = {}
            if solvers is not []:
                for solver in solvers:
                    if solver == 'gaussian':
                        print("@>: computing lmi correlation matrix")
                        MIdict.update({'gcc_lmi': aux.compute_generalized_correlation_coefficients(values,
                                                                                                   features_dimension=len(
                                                                                                       self.dih_labels),
                                                                                                   solver=solver,
                                                                                                   correction=False)})
                        if CENTRALITY:
                            print("@>: computing eigenvector centrality from lmi matrix")
                            _, ec = aux.compute_eigenvector_centrality(MIdict['gcc_lmi'], weight='weight')
                            ECdict.update({'gcc_lmi': ec})

                    elif 'knn' in solver:
                        print("@>: computing mi correlation matrix")
                        MIdict.update({'gcc_mi': aux.compute_generalized_correlation_coefficients(values,
                                                                                                  features_dimension=len(
                                                                                                      self.dih_labels),
                                                                                                  solver=solver,
                                                                                              correction=True)})
                        if CENTRALITY:
                            print("@>: computing eigenvector centrality from mi matrix")
                            _, ec = aux.compute_eigenvector_centrality(MIdict['gcc_mi'], weight='weight')
                            ECdict.update({'gcc_mi': ec})

                self.dih_gcc_allreplicas['rep_%d' % win_idx] = MIdict
                self.dih_eigen_centrality_allreplicas['rep_%d' % win_idx] = ECdict

            if COV_DISP:
                print("@>: computing covariance of dihedral fluctuations")
                self.covar_dih_allreplicas['rep_%d' % win_idx] = aux.evaluate_covariance_matrix(values, center=center)

            if PCC:
                print("@>: computing square displacements")
                self.disp_from_mean_dih_allreplicas['rep_%d' % win_idx] = aux._compute_square_displacements(values)

                print("@>: computing pearson correlation matrix")
                self.dih_pcc_allreplicas['rep_%d' % win_idx] = \
                    np.corrcoef(self.disp_from_mean_dih_allreplicas['rep_%d' % win_idx], rowvar=False)

            if DCC:
                print("@>: computing DCC")
                values_reshaped = values.reshape((values.shape[0], values.shape[1] * values.shape[2]))
                values_ave = np.mean(values_reshaped, axis=0)
                self.dih_dcc_allreplicas['rep_%d' % win_idx] = aux.compute_DCC(values_reshaped, features_dim,
                                                                               normalized=normalized)
