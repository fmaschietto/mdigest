"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: fmaschietto, bcallen95"""

from mdigest.core.imports            import *
from MDAnalysis.analysis       import distances
from concurrent.futures.thread import ThreadPoolExecutor
import mdigest.core.savedata            as sd
import mdigest.core.toolkit              as tk
import mdigest.core.auxiliary  as aux


MAX_WORKERS = 4

class KS_Box:
    """
    Use as collector to free up memory
    """
    def __init__(self, KS, attrlist):
        import gc
        for attr in list(KS.__dict__().keys()):
            if attr in attrlist:
                setattr(self, attr, KS.__dict__()[attr])
        for v in dir(KS):
            if not v.startswith('__'):
                del v
        gc.collect()
        del KS



class KS_Energy:
    """General purpose class handling computation Kabsch-Sander Analysis over MD trajectories."""

    def __init__(self, MDSIM):
        """
        Description
        -----------
        General purpose class handling computation Kabsch-Sander Analysis over MD trajectories.

        Parameters
        ----------
        MDSIM: class object

        Attributes
        ----------
        self.indices: dict,
            maps a list of backbone atom label strings to a list of indices corresponding to the index of the selected
            backbone atom for each residue
        self.backbone_dictionary: dict,
            maps a list of backbone atom label strings to a list corresponding atom names (as found in the topology)
        self.is_protein: np.ndarray of dtype bool and shape (number of selected residues (protein or not))
            is_protein array is 1 if the corresponding residue has C,O,N,H protein backbone atoms in the topology, 0 otherwise
        self.is_proline: np.ndarray of dtype bool and shape (nresidues),
            is_protein array is 1 if the corresponding residue is a proline, 0 otherwise
        self.offset: int,
            default offset is 0, adjust if the resindices of the selcted atoms do not start at 0
        self.q1q2: np.ndarray of shape (nresidues, nresidues),
            entries ($q1q2_(ij)$)) are the products of the i-th CO and j-th NH residue charges extracted
            from the topology
        self.bb_distances_allrep: dict,
            stores distances between specified backbone atoms of each pair or residues used in KS
            calculation such as CN, ON, OH, CH distance matrices for each replica,
                - example: ``bb_distances_allrep['rep_0]['CN']`` returns a np.ndarray of shape (nresidues*nresidues) containing all pairwise C-N distances
        self.KS_energies_allrep: dict,
            stores KS energies for each replica of a given simulation object in the form of np.ndarray
            of shape (window_span, n_nresidues, nresidues),
                - example: access as KS_energies_allrep['rep_0] for KS energies of replica 0
        self.KS_DA_energies_allrep: dict,
            stores KS energies summed over columns(acceptor)/rows(donor), yielding acceptor/donor
            KS energy matrices of shape (window_span, n_nresidues)
                - example:
                    1) access as ``KS_DA_energies_allrep['rep_0']['donor']`` for donor energies of replica 0
                    2) access as ``KS_DA_energies_allrep['rep_0']['acceptor']`` for acceptor energies of replica 0
        self.KS_DA_LMI_corr_allrep: dict,
            stores per replica don_acc mutual information based generalized correlation matrices computed respectively from donor, acceptor and donor+acceptor KS energies
                - example:
                    access as ``KS_DA_MI_corr_allrep['rep_0']['don_acc']``
        self.KS_DA_LMI_corr_allrep: dict,
            stores per replica donor/acceptor/don_acc linearized mutual information based generalized correlation matrices computed respectively from donor, acceptor and donor+acceptor KS energies
                - example:
                    access as ``KS_DA_LMI_corr_allrep['rep_0']['donor']``
        self.KS_cov_allrep: dict
            stores per replica covariance matrix over donor, acceptor KS energy matrices computed along the trajectories
                - example:
                    access as ``KS_cov_allrep['rep_0']['donor']`` for the covariance of the donor KS energies ``KS_cov_allrep['rep_0']['acceptor']`` for the covariance of the acceptor KS energies
        self.eigvec_centrality_da_allrep: dict,
            stores the eigvector centrality array for each replica (rep) computed diagonalizing the ``KS_DA_LMI_corr_allrep[rep]['don_acc']`` matrix
        self.eigvec_centrality_don_allrep: dict,
            stores the eigvector centrality array for each replica (rep) computed diagonalizing the ``KS_DA_LMI_corr_allrep[rep]['donor']`` matrix
        self.eigvec_centrality_acc_allrep: dict,
            stores the eigvector centrality array for each replica (rep) computed diagonalizing the
            ``KS_DA_LMI_corr_allrep[rep]['acceptor']`` matrix
        self.eigvec_centrality_da_indep_sum_allrep: dict,
            stores the eigvector centrality array for each replica (rep) computed diagonalizing independently the ``KS_DA_LMI_corr_allrep[rep]['donor']`` and
            ``KS_DA_LMI_corr_allrep[rep]['acceptor']`` matrix and summing the 1st eigenvectors together

        Methods
        -------

        References
        ----------
        """
        self.mds_data                              = MDSIM.mds_data
        self.mda_u                                 = MDSIM.mda_u
        self.num_replicas                          = MDSIM.num_replicas
        self.initial                               = MDSIM.initial
        self.final                                 = MDSIM.final
        self.step                                  = MDSIM.step
        self.window_span                           = MDSIM.window_span
        self.atom_group_selstr                     = MDSIM.atom_group_selstr
        self.system_selstr                         = MDSIM.system_selstr
        self.nframes_per_replica                   = MDSIM.nframes_per_replica
        self.atom_group_selection                  = MDSIM.atom_group_selection
        self.nresidues                             = MDSIM.nresidues
        self.natoms                                = MDSIM.natoms
        self.total_nframes                         = MDSIM.total_nframes
        self.nodes_to_res_dictionary               = MDSIM.nodes_to_res_dictionary # useful for later plotting (match nodes id to resid)

        self.indices                               = None
        self.backbone_dictionary                   = None
        self.is_protein                            = None
        self.is_proline                            = None
        self.offset                                = 0
        self.q1q2                                  = None

        self.bb_distances_allrep                   = {}
        self.KS_energies_allrep                    = {}
        self.KS_DA_energies_allrep                 = {}
        self.KS_DA_MI_corr_allrep                  = {}
        self.KS_DA_LMI_corr_allrep                 = {}
        self.KS_cov_allrep                         = {}
        self.eigvec_centrality_da_allrep           = {}
        self.eigvec_centrality_don_allrep          = {}
        self.eigvec_centrality_acc_allrep          = {}
        self.eigvec_centrality_da_mean_allrep       = {}
        self.eigvec_centrality_da_indep_sum_allrep = {}


    def save_class(self, file_name_root='./output/cache/', save_space=False):
        """
        Save MDS class instances to file

        Parameters
        ----------
        file_name_root: str,
            path where to save class
        save_space: bool,
            if False bb_distances_allrep and KS_energies_allrep are not dumped to file
        """

        self.mds_data = sd.MDSdata()
        self.mds_data.is_protein                            = self.is_protein
        self.mds_data.is_proline                            = self.is_proline
        self.mds_data.q1q2                                  = self.q1q2
        self.mds_data.nodes_to_res_dictionary               = self.nodes_to_res_dictionary
        self.mds_data.bb_distances_allrep                   = self.bb_distances_allrep
        self.mds_data.KS_energies_allrep                    = self.KS_energies_allrep
        self.mds_data.KS_DA_energies_allrep                 = self.KS_DA_energies_allrep
        self.mds_data.KS_DA_MI_corr_allrep                  = self.KS_DA_MI_corr_allrep
        self.mds_data.KS_DA_LMI_corr_allrep                 = self.KS_DA_LMI_corr_allrep
        self.mds_data.KS_cov_allrep                         = self.KS_cov_allrep
        self.mds_data.eigvec_centrality_da_allrep           = self.eigvec_centrality_da_allrep
        self.mds_data.eigvec_centrality_don_allrep          = self.eigvec_centrality_don_allrep
        self.mds_data.eigvec_centrality_acc_allrep          = self.eigvec_centrality_acc_allrep
        self.mds_data.eigvec_centrality_da_mean_allrep       = self.eigvec_centrality_da_mean_allrep
        self.mds_data.eigvec_centrality_da_indep_sum_allrep = self.eigvec_centrality_da_indep_sum_allrep
        self.mds_data.save_to_file(file_name_root, save_space=save_space)


    def set_selection(self, atom_group_selstr, system_selstr='all'):
        """
        Set selection strings

        Parameters
        ----------
        system_selstr: str,
            selection string to be used for extracting a subset of the atoms (system) on which to perform analysis

        atom_group_selstr: str,
            selection string to be used for selecting a subset of atoms from the system
                - atom_str_sel: a list of four selection strings containing in order the N, O, C, H backbone selection strings, respectively.)

        Examples
        --------
        ``KS_Energy.set_selection(['protein and backbone and name N','protein and backbone and name O', 'protein and backbone and name C','protein and name H'], system_selstr='protein')``

        """
        keys = ['N-Backbone', 'O-Backbone', 'C-Backbone', 'H-Backbone']

        self.atom_group_selstr = dict(zip(keys, atom_group_selstr))
        self.system_selstr = system_selstr


    def set_charges_array(self, chargeOtimeschargeN):
        """
        Set charges array

        Parameters
        ----------
        chargeOtimeschargeN: np.ndarray,
            array of dimensions (nresidues, nresidues) entries (q1q2$_(ij)$)) are the products of the i-th CO and j-th NH residue charges extracted from the topology
        """
        self.q1q2 = chargeOtimeschargeN


    def set_offset(self, offset):
        """
        set offset

        Parameters
        ----------

        offset: int,
            set offset when the residue indexes in the topology start at number other than 0
            integer that should be subtracted to the first residue index have the resindices list start from 0
        """
        self.offset = offset


    def set_backbone_dictionary(self, backbone_dictionary):
        """
        Set backbone dictionary

        Parameters
        ----------

        backbone_dictionary: dict,
            backbone dictionary specifying the atom name of each backbone atom.
            Names should match those in the topology files

        Examples
        --------
        ``KS_energy.set_backbone_dictionary({'N-Backbone':'N', 'O-Backbone':'O','C-Backbone':'C', 'CA-Backbone':'CA', 'H-Backbone':'H'})``

        """
        self.backbone_dictionary = backbone_dictionary


    def _distances(self, indices1, indices2, beg, end, stride, remap=False):
        """
        Compute distances
        """
        count = 0
        d = np.zeros((self.nframes_per_replica, len(indices1), len(indices2)), dtype=float)
        for ts in self.mda_u.trajectory[beg:end:stride]:
            coordinates = ts.positions
            d[count] = distances.distance_array(coordinates[indices1], coordinates[indices2], backend="OpenMP")
            if not remap:
                d[count][:, np.where(self.is_protein == 0)[0]] = 0.

            count += 1
        return d


    # def compute_distances_parallel(self, beg, end, stride, remap=False):
    #     """
    #     Compute distances in parallel
    #
    #     Parameters
    #     ----------
    #
    #     beg: int,
    #         initial frame
    #     end: int,
    #         end frame
    #     stride: int,
    #         step
    #     remap: bool,
    #         if True assumes remapping is unneeded (all distance matrices have the same dimension)
    #
    #     Returns
    #     -------
    #     bb_dist_dict: dict
    #         dictionary with CN, CH, OH, ON as keys and np.ndarrays with the corresponding distance arrays as values
    #     """
    #
    #     print('@>: computing distances in parallel')
    #
    #     C = self.indices['C-Backbone']
    #     N = self.indices['N-Backbone']
    #     O = self.indices['O-Backbone']
    #     H = self.indices['H-Backbone']
    #
    #     func = self._distances
    #     calls = { 'CN': func(C, N, beg, end, stride, remap=True),
    #               'CH': func(C, H, beg, end, stride, remap=remap),
    #               'OH': func(O, H, beg, end, stride, remap=remap),
    #               'ON': func(O, N, beg, end, stride, remap=True)}
    #
    #     processes = []
    #     results = {}
    #     with ThreadPoolExecutor(MAX_WORKERS) as executor:
    #         for key, obj_ in calls.items():
    #             processes.append(executor.submit(obj_))
    #             results.update({key:obj_})
    #
    #     bb_dist_dict = {}
    #
    #     bb_dist_dict.update({'CH': results['CH']})
    #     bb_dist_dict.update({'ON': results['ON']})
    #     bb_dist_dict.update({'OH': results['OH']})
    #     bb_dist_dict.update({'CN': results['CN']})
    #     return bb_dist_dict

    ############# NOTES #########################

    # The processes list is created to store the references to the executor.submit objects returned by the ThreadPoolExecutor.
    # However, these references are not used later in the code. As a result, the garbage collector may not be able to release the memory
    # associated with these objects. To address this issue, we can remove the processes list,
    # allowing the objects to be automatically garbage collected.
    #
    # To fix these issues, you can modify the compute_distances_parallel method as follows:
    # TEST IT !


    def compute_distances_parallel(self, beg, end, stride, remap=False):
        """
        Compute distances in parallel

        Parameters
        ----------
        beg: int,
            initial frame
        end: int,
            end frame
        stride: int,
            step
        remap: bool,
            if True assumes remapping is unneeded (all distance matrices have the same dimension)

        Returns
        -------
        bb_dist_dict: dict
            dictionary with CN, CH, OH, ON as keys and np.ndarrays with the corresponding distance arrays as values
        """

        print('@>: computing distances in parallel')

        C = self.indices['C-Backbone']
        N = self.indices['N-Backbone']
        O = self.indices['O-Backbone']
        H = self.indices['H-Backbone']

        func = self._distances

        results = {}
        with ThreadPoolExecutor(MAX_WORKERS) as executor:
            results['CN'] = executor.submit(func, C, N, beg, end, stride, remap=True)
            results['CH'] = executor.submit(func, C, H, beg, end, stride, remap=remap)
            results['OH'] = executor.submit(func, O, H, beg, end, stride, remap=remap)
            results['ON'] = executor.submit(func, O, N, beg, end, stride, remap=True)

        bb_dist_dict = {}

        bb_dist_dict.update({'CH': results['CH'].result()})
        bb_dist_dict.update({'ON': results['ON'].result()})
        bb_dist_dict.update({'OH': results['OH'].result()})
        bb_dist_dict.update({'CN': results['CN'].result()})
        return bb_dist_dict



    def compute_KS_energy(self, dist_dict, topology_charges=False):
        """
        Perform KS calculation

        Parameters
        ----------
        dist_dict: dict,
            single dictionary containing residue-to-residues backbone atom distances for a given replica
        topology_charges: bool,
            if True, ``self.q1q2`` is expected to be filled with charges array

        Returns
        -------
        KS_energies: np.ndarray,
            KS_energies
        """

        CH_dist = dist_dict['CH']
        ON_dist = dist_dict['ON']
        OH_dist = dist_dict['OH']
        CN_dist = dist_dict['CN']

        nframes = len(CN_dist)
        nresidues = len(CN_dist[0])
        KS_energies = np.zeros((nframes, nresidues, nresidues), dtype=np.float32)
        if not topology_charges:
            q1timesq2 = 0.42 * 0.20
        else:
            q1timesq2 = self.q1q2
        ffactor = 332
        for frame in trange(nframes):

            inv_ON = np.reciprocal(ON_dist[frame])
            inv_CH = np.reciprocal(CH_dist[frame])
            inv_OH = np.reciprocal(OH_dist[frame])
            inv_CN = np.reciprocal(CN_dist[frame])
            inv_OH = np.nan_to_num(inv_OH, nan=0.0, posinf=0.0, neginf=0.0)
            inv_CN = np.nan_to_num(inv_CN, nan=0.0, posinf=0.0, neginf=0.0)
            inv_CH = np.nan_to_num(inv_CH, nan=0.0, posinf=0.0, neginf=0.0)
            inv_ON = np.nan_to_num(inv_ON, nan=0.0, posinf=0.0, neginf=0.0)

            KS_energies[frame, :, :] = q1timesq2 * (inv_ON + inv_CH - inv_OH - inv_CN) * ffactor
        return KS_energies


    def prepare_kabsch_sander_arrays(self):
        """
        Prepare Kabsch-Sanders calculation
        """

        C_str  = self.backbone_dictionary['C-Backbone']
        CA_str = self.backbone_dictionary['CA-Backbone']
        N_str  = self.backbone_dictionary['N-Backbone']
        O_str  = self.backbone_dictionary['O-Backbone']
        H_str  = self.backbone_dictionary['H-Backbone']

        u = self.mda_u

        ca_indices, n_indices, c_indices, o_indices, h_indices, is_proline, is_protein = \
            [], [], [], [], [], [], []
        for residue in u.select_atoms(self.system_selstr).residues:


            ca = tk.get_or_minus1(lambda: [a.index for a in residue.atoms if a.name == CA_str][0])
            h  = tk.get_or_minus1(lambda: [a.index for a in residue.atoms if a.name == H_str][0])
            n  = tk.get_or_minus1(lambda: [a.index for a in residue.atoms if a.name == N_str][0])
            c  = tk.get_or_minus1(lambda: [a.index for a in residue.atoms if a.name == C_str][0])
            o  = tk.get_or_minus1(lambda: [a.index for a in residue.atoms if a.name == O_str][0])

            ca_indices.append(ca)
            n_indices.append(n)
            c_indices.append(c)
            o_indices.append(o)
            h_indices.append(h)

            is_proline.append(residue.resname == 'PRO')
            is_protein.append(h != -1 and n != -1 and c != -1 and o != -1)

        ca_indices  = np.array(ca_indices, np.int32)
        n_indices = np.array(n_indices, np.int32)
        c_indices = np.array(c_indices, np.int32)
        o_indices = np.array(o_indices, np.int32)
        h_indices = np.array(h_indices, np.int32)

        # ca_indices[ca_indices == -1] = ca_indices[ca_indices  == -1]
        n_indices[n_indices  == -1] = ca_indices[n_indices  == -1]
        c_indices[c_indices  == -1] = ca_indices[c_indices  == -1]
        o_indices[o_indices  == -1] = ca_indices[o_indices  == -1]
        h_indices[h_indices  == -1] = ca_indices[h_indices  == -1]

        is_proline = np.array(is_proline, np.int32)
        is_protein = np.array(is_protein, np.int32)

        self.indices = dict(zip(sorted(list(self.backbone_dictionary.keys())),
                                [c_indices, ca_indices, h_indices, n_indices, o_indices]))

        self.is_protein = is_protein
        self.is_proline = is_proline



    def compute_EEC(self, distance_matrix=None, loc_factor=None, don_acc=True):
        """
        Compute Electrostatic Eigenvector Centrality (EEC), donor/acceptor/don_acc centrality for each replica

        Parameters
        ----------

        distance_matrix: default None,
            provide distance matrix when loc_factor != 0 to zero out values
            adiacency matrix values corresponding to distances exceeding loc_factor
        loc_factor: float,
            filtering threshold for selection of specific correlation range
        don_acc: bool,
            whether to compute DA (donor_acceptor) and D+A (donor+acceptor) centralities
        """

        num_replicas = self.num_replicas
        ec_don_acc = None
        ec_don_acc_mean = None

        for win_idx in tk.log_progress(range(num_replicas), every=1, size=num_replicas, name="Window"):

            if loc_factor is not None:
                _, ec_donor = aux.compute_eigenvector_centrality(self.KS_DA_LMI_corr_allrep['rep_%d' %win_idx]['donor'], loc_factor=loc_factor,
                                                          distmat=distance_matrix)
                _, ec_acceptor = aux.compute_eigenvector_centrality(self.KS_DA_LMI_corr_allrep['rep_%d' %win_idx]['acceptor'], loc_factor=loc_factor,
                                                          distmat=distance_matrix)
                if don_acc:
                    _, ec_don_acc = aux.compute_eigenvector_centrality(self.KS_DA_LMI_corr_allrep['rep_%d' %win_idx]['don_acc'], loc_factor=loc_factor,
                                                          distmat=distance_matrix)
                    _, ec_don_acc_mean = aux.compute_eigenvector_centrality(self.KS_DA_LMI_corr_allrep['rep_%d' %win_idx]['mean_don_acc'], loc_factor=loc_factor,
                                                          distmat=distance_matrix)


            else:
                _, ec_donor = aux.compute_eigenvector_centrality(self.KS_DA_LMI_corr_allrep['rep_%d' %win_idx]['donor'], loc_factor=None,
                                                          distmat=None)
                _, ec_acceptor = aux.compute_eigenvector_centrality(self.KS_DA_LMI_corr_allrep['rep_%d' %win_idx]['acceptor'], loc_factor=None,
                                                          distmat=None)
                if don_acc:
                    _, ec_don_acc = aux.compute_eigenvector_centrality(self.KS_DA_LMI_corr_allrep['rep_%d' % win_idx]['don_acc'],
                                                          loc_factor=None, distmat=None)
                    _, ec_don_acc_mean = aux.compute_eigenvector_centrality(self.KS_DA_LMI_corr_allrep['rep_%d' %win_idx]['mean_don_acc'], loc_factor=loc_factor,
                                                          distmat=distance_matrix)

            ec_independent_sum = ec_donor + ec_acceptor
            # ec = np.array([*ec.values()], dtype=object)

            if don_acc:
                # from gcc joint gcc [don,acc]
                self.eigvec_centrality_da_allrep.update({'rep_%d' %win_idx: ec_don_acc})
                # from gcc gccdon + gccacc
                self.eigvec_centrality_da_mean_allrep.update({'rep_%d' % win_idx: ec_don_acc_mean})


            self.eigvec_centrality_don_allrep.update({'rep_%d' %win_idx: ec_donor})
            self.eigvec_centrality_acc_allrep.update({'rep_%d' %win_idx: ec_acceptor})
            self.eigvec_centrality_da_indep_sum_allrep.update({'rep_%d' %win_idx: ec_independent_sum})


    def KS_pipeline(self, topology_charges=False, covariance=False, MI=None, **kwargs):
        """
        KS Pipeline

        Parameters
        ----------
        topology_charges: bool,
            whether to use topology charges in KS calculation
        covariance: bool,
            whether to compute covariance of KS_energies
        MI: str or None,
            - if None skip computation of MI based correlation
            - if 'knn_arg1_arg2' compute MI using k=arg1, and estimator=arg2; default is 'knn_5_1'
        """

        def _compress_KS_energies(KS_E):
            """
            Compress KS_energies into donor and acceptor energies
            """
            KSenergy_acc = np.sum(KS_E, axis=1)
            KSenergy_don = np.sum(KS_E, axis=2)
            KS_energy_dict = {}
            KS_energy_dict.update({'donor': KSenergy_don,
                                   'acceptor': KSenergy_acc})
            return KS_energy_dict

        def _compute_KS_correlation(KS_energy_dict):
            """
            Compute Linearized mutual information based correlation of donor and acceptor KS energies using gaussian
            estimator to evaluate mutual information.
            """


            LMI_corr_donor = aux.compute_generalized_correlation_coefficients(KS_energy_dict['donor'],
                                                                              features_dimension=1, solver='gaussian',
                                                                              correction=False,
                                                                              subset=None)
            LMI_corr_acceptor = aux.compute_generalized_correlation_coefficients(KS_energy_dict['acceptor'],
                                                                              features_dimension=1, solver='gaussian',
                                                                              correction=False,
                                                                              subset=None)
            LMI_KS_corr = aux.compute_generalized_correlation_coefficients(KS_energy_dict['acceptor']+KS_energy_dict['donor'],
                                                                              features_dimension=1, solver='gaussian',
                                                                              correction=False,
                                                                              subset=None)


            LMI_KS_corr_sum = 0.5 * (LMI_corr_donor + LMI_corr_acceptor)
            LMI_KS_corr_dict = {}
            LMI_KS_corr_dict.update({'donor':    LMI_corr_donor,
                                     'acceptor': LMI_corr_acceptor,
                                     'don_acc':  LMI_KS_corr,
                                     'mean_don_acc':  LMI_KS_corr_sum})
            return LMI_KS_corr_dict

        def _compute_covar_KS_energies(KS_energy_dict):
            """
            Compute covariance of KS energies
            """
            cov_donor    = np.cov(KS_energy_dict['donor'],    rowvar=False, bias=True)
            cov_acceptor = np.cov(KS_energy_dict['acceptor'], rowvar=False, bias=True)
            # cov_don_acc  = np.cov(KS_energy_dict['don_acc'],  rowvar=False, bias=True)
            KS_cov_dict = {}
            KS_cov_dict.update({'donor': cov_donor})
            KS_cov_dict.update({'acceptor': cov_acceptor})
            # KS_cov_dict.update({'don_acc': cov_don_acc})
            return KS_cov_dict

        print('@>: prepare kabsch sanders calculation')
        self.prepare_kabsch_sander_arrays()
        print('@>: run KS calculation')


        for win_idx in tk.log_progress(range(self.num_replicas), every=1, size=self.num_replicas, name="Window"):
            #beg =  int(self.final/self.num_replicas)* win_idx
            #end =  int(self.final/self.num_replicas)* (win_idx + 1)

            offset =  (self.final - self.initial)// self.num_replicas
            if self.window_span != offset/self.step:
                print("@>: WARNING: the offset is not equal to the window span")

            beg = self.initial + offset * win_idx
            end = self.initial + offset * (win_idx + 1)

            print("@>: KS energy calculation ...")
            print("@>: begin frame: %d" % beg)
            print("@>: end   frame: %d" % end)
            print("@>: step:        %d" % self.step)
            stride = self.step
            bb_distances_dict = self.compute_distances_parallel(beg, end, stride)
            KS_energies = self.compute_KS_energy(bb_distances_dict, topology_charges=topology_charges)
            self.bb_distances_allrep.update({'rep_%d' % win_idx: bb_distances_dict})
            self.KS_energies_allrep.update({'rep_%d'  % win_idx: KS_energies})
            KS_DA_energies  = _compress_KS_energies(KS_energies)
            self.KS_DA_energies_allrep.update({'rep_%d' % win_idx: KS_DA_energies})
            KS_DA_LMI_corr = _compute_KS_correlation(KS_DA_energies)
            self.KS_DA_LMI_corr_allrep.update({'rep_%d' % win_idx: KS_DA_LMI_corr})

            # setup kwargs
            if MI:
                try:
                    correction = kwargs['correction']
                except KeyError:
                    correction = True

                try:
                    subset = kwargs['subset']
                except KeyError:
                    subset = None

                if MI is not None:
                    solver = MI
                else:
                    print('@>: default solver set to knn_5_1')
                    solver = 'knn_5_1'

                KS_DA_MI_corr = aux.compute_generalized_correlation_coefficients(
                    KS_DA_energies['acceptor'] + KS_DA_energies['donor'],
                    features_dimension=1, solver=solver, correction=correction, subset=subset)

                self.KS_DA_MI_corr_allrep.update({'rep_%d' % win_idx: {'don_acc': KS_DA_MI_corr}})
            if covariance:
                KS_cov = _compute_covar_KS_energies(KS_DA_energies)
                self.KS_cov_allrep.update({'rep_%d' % win_idx: KS_cov})

