"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: fmaschietto, bcallen95"""

from silx.io.dictdump import dicttoh5
from silx.io.dictdump import load as dictload
from mdigest.core.toolkit import *
from mdigest.core.imports import *

ds_args = {'compression': "gzip",
           'shuffle': True,
           'fletcher32': True}


class MDSdata:
    """
    Save insances from mdigest.DynCorr, mdigest.DihDynCorr, mdigest.KS_Energy,
    mdigest.CMTY for easy access.
    ``[**]`` function structure adapted from https://github.com/melomcr/dynetan
    """
    def __init__(self):
        self.nodes_to_res_dictionary = None
        self.nodes_dih_to_indices_dictionary = None

        # attributes from correlation.DynCorr
        self.nresidues = None
        self.nnodes = None
        self.nodes_idx_array = None
        self.coordinates_allreplicas = None
        self.displacements_allreplicas = None
        self.distances_allreplicas = None
        self.covariances_allreplicas = None
        self.covar_disp_allreplicas = None
        self.lmi_allreplicas = None
        self.gcc_allreplicas = None
        self.dcc_allreplicas = None
        self.pcc_allreplicas = None
        self.atom_group_selstr = None
        self.system_selstr = None
        self.exclusion_matrix_allreplicas = None
        self.eigenvector_centrality_allreplicas = None

        # attributes from dcorrelation.DihDynCorr
        self.dih_labels = None
        self.dih_indices = None
        self.dihedrals_allreplicas = None
        self.disp_from_mean_dih_allreplicas = None
        self.covar_dih_allreplicas = None
        self.dih_gcc_allreplicas = None
        self.dih_dcc_allreplicas = None
        self.dih_pcc_allreplicas = None
        self.dih_eigen_centrality_allreplicas = None

        # attributes from kscorrelation.KS_energy
        self.bb_distances_allrep = None
        self.KS_energies_allrep = None
        self.KS_DA_energies_allrep = None
        self.KS_DA_MI_corr_allrep = None
        self.KS_DA_LMI_corr_allrep = None
        self.KS_cov_allrep = None

        self.is_protein = None
        self.is_proline = None
        self.q1q2 = None

        self.eigvec_centrality_da_allrep = None
        self.eigvec_centrality_don_allrep = None
        self.eigvec_centrality_acc_allrep = None
        self.eigvec_centrality_da_mean_allrep = None
        self.eigvec_centrality_da_indep_sum_allrep = None

        # attributes from network_communities.CMTY

        self.nodes_communities_collect = None
        self.predecessors = None
        self.nxGraphs_dict = None

        self.partitions_collect = None
        self.distances_collect = None
        self.max_distance = None
        self.max_direct_distance = None

        #  attributes not saved to file but are reconstructed from the loaded information
        self.atom_group_selection = None
        self.num_replicas = None

    def save_to_file(self, file_name_root, save_space=False):
        """
        Opens the HDF5 file and stores all data

        Parameters
        ----------
        file_name_root: str,
            file rootname
        save_space: bool,
            if set to True avoid saving to file some very large attributes.
        """

        with h5py.File(file_name_root + "_cache.hf", "w") as f:

            # atomic displacements correlation related attributes (arrays)
            if self.atom_group_selstr is not None:
                f_atom_group_selstr = f.create_dataset("atom_group_selstr",
                                                       shape=np.asarray(self.atom_group_selstr).astype('S100').shape,
                                                       dtype=np.asarray(self.atom_group_selstr).astype('S100').dtype,
                                                       data=np.asarray(self.atom_group_selstr).astype('S100'))
            if self.system_selstr is not None:
                f_system_selstr = f.create_dataset("system_selstr",
                                                   shape=np.asarray(self.system_selstr).astype('S100').shape,
                                                   dtype=np.asarray(self.system_selstr).astype('S100').dtype,
                                                   data=np.asarray(self.system_selstr).astype('S100'))

            if self.nodes_idx_array is not None:
                f_nodes_idx_array = f.create_dataset("nodes_idx_array",
                                                     shape=self.nodes_idx_array.shape,
                                                     dtype=self.nodes_idx_array.dtype,
                                                     data=self.nodes_idx_array)

            # dihedral correlation related attributes (arrays)
            if self.dih_labels is not None:
                self.dih_labels = np.asarray(self.dih_labels, dtype='S10')
                f_dih_labels = f.create_dataset("dih_labels",
                                                shape=self.dih_labels.shape,
                                                dtype=self.dih_labels.dtype,
                                                data=self.dih_labels)

            if self.dih_indices is not None:
                self.dih_indices = np.asarray(self.dih_indices, dtype=int)
                f_dih_indices = f.create_dataset("dih_indices",
                                                 shape=self.dih_indices.shape,
                                                 dtype=self.dih_indices.dtype,
                                                 data=self.dih_indices)

            # KS_energy related attributes (arrays)
            if self.is_protein is not None:
                f_is_protein = f.create_dataset("is_protein",
                                                shape=self.is_protein.shape,
                                                dtype=self.is_protein.dtype,
                                                data=self.is_protein)
            if self.is_proline is not None:
                f_is_proline = f.create_dataset("is_proline",
                                                shape=self.is_proline.shape,
                                                dtype=self.is_proline.dtype,
                                                data=self.is_proline)
            if self.q1q2 is not None:
                f_q1q2 = f.create_dataset("q1q2",
                                          shape=self.q1q2.shape,
                                          dtype=self.q1q2.dtype,
                                          data=self.q1q2)

            #  communities related attributes

            if self.max_direct_distance is not None:
                self.max_direct_distance = np.asarray([self.max_direct_distance])
                f_max_direct_distance = f.create_dataset("max_direct_distance",
                                                         shape=self.max_direct_distance.shape,
                                                         dtype=self.max_direct_distance.dtype,
                                                         data=self.max_direct_distance)

            if self.max_distance is not None:
                self.max_distance = np.asarray([self.max_distance])
                f_max_distance = f.create_dataset("max_distance",
                                                  shape=self.max_distance.shape,
                                                  dtype=self.max_distance.dtype,
                                                  data=self.max_distance)

                # atomic coordinate correlation  related attributes (dictionaries)
        if self.nodes_to_res_dictionary is not None:
            dicttoh5(self.nodes_to_res_dictionary, file_name_root + '_nodes_to_res_dictionary.h5',
                     create_dataset_args=ds_args)

        if self.nodes_dih_to_indices_dictionary is not None:
            dicttoh5(self.nodes_dih_to_indices_dictionary, file_name_root + '_nodes_dih_to_indices_dictionary.h5',
                     create_dataset_args=ds_args)

        if self.coordinates_allreplicas is not None:
            dicttoh5(self.coordinates_allreplicas, file_name_root + '_coordinates_allreplicas.h5',
                     create_dataset_args=ds_args)

        if self.displacements_allreplicas is not None:
            dicttoh5(self.displacements_allreplicas, file_name_root + '_displacements_allreplicas.h5',
                     create_dataset_args=ds_args)

        if self.distances_allreplicas is not None:
            dicttoh5(self.distances_allreplicas, file_name_root + '_distances_allreplicas.h5',
                     create_dataset_args=ds_args)

        if self.covar_disp_allreplicas is not None:
            dicttoh5(self.covar_disp_allreplicas, file_name_root + '_covar_disp_allreplicas.h5',
                     create_dataset_args=ds_args)

        if self.gcc_allreplicas is not None:
            dicttoh5(self.gcc_allreplicas, file_name_root + '_gcc_allreplicas.h5',
                     create_dataset_args=ds_args)

        if self.dcc_allreplicas is not None:
            dicttoh5(self.dcc_allreplicas, file_name_root + '_dcc_allreplicas.h5',
                     create_dataset_args=ds_args)

        if self.pcc_allreplicas is not None:
            dicttoh5(self.pcc_allreplicas, file_name_root + '_pcc_allreplicas.h5',
                     create_dataset_args=ds_args)

        if self.exclusion_matrix_allreplicas is not None:
            dicttoh5(self.exclusion_matrix_allreplicas, file_name_root + '_exclusion_matrix_allreplicas.h5',
                     create_dataset_args=ds_args)

        if self.eigenvector_centrality_allreplicas is not None:
            dicttoh5(self.eigenvector_centrality_allreplicas, file_name_root + '_eigenvector_centrality_allreplicas.h5',
                     create_dataset_args=ds_args)

        # dihedral correlation related attributes (dictionaries)
        if self.dihedrals_allreplicas is not None:
            dicttoh5(self.dihedrals_allreplicas, file_name_root + '_dihedrals_allreplicas.h5',
                     create_dataset_args=ds_args)

        if self.disp_from_mean_dih_allreplicas is not None:
            dicttoh5(self.disp_from_mean_dih_allreplicas, file_name_root + '_disp_from_mean_dih_allreplicas.h5',
                     create_dataset_args=ds_args)

        if self.covar_dih_allreplicas is not None:
            dicttoh5(self.covar_dih_allreplicas, file_name_root + '_covar_dih_allreplicas.h5',
                     create_dataset_args=ds_args)

        if self.dih_dcc_allreplicas is not None:
            dicttoh5(self.dih_dcc_allreplicas, file_name_root + '_dih_dcc_allreplicas.h5', create_dataset_args=ds_args)

        if self.dih_pcc_allreplicas is not None:
            dicttoh5(self.dih_pcc_allreplicas, file_name_root + '_dih_pcc_allreplicas.h5', create_dataset_args=ds_args)

        if self.dih_gcc_allreplicas is not None:
            dicttoh5(self.dih_gcc_allreplicas, file_name_root + '_dih_gcc_allreplicas.h5', create_dataset_args=ds_args)

        if self.dih_eigen_centrality_allreplicas is not None:
            dicttoh5(self.dih_eigen_centrality_allreplicas, file_name_root + '_dih_eigen_centrality_allreplicas.h5',
                     create_dataset_args=ds_args)

        # KS_energy related attributes (dictionaries)
        if self.KS_energies_allrep is not None:
            if not save_space:
                dicttoh5(self.bb_distances_allrep, file_name_root + '_bb_distances_allrep.h5',
                         create_dataset_args=ds_args)
                dicttoh5(self.KS_energies_allrep, file_name_root + '_energies_allrep.h5', create_dataset_args=ds_args)
        if self.KS_DA_energies_allrep is not None:
            dicttoh5(self.KS_DA_energies_allrep, file_name_root + '_DA_energies_allrep.h5', create_dataset_args=ds_args)
        if self.KS_DA_LMI_corr_allrep is not None:
            dicttoh5(self.KS_DA_LMI_corr_allrep, file_name_root + '_DA_LMI_corr_allrep.h5', create_dataset_args=ds_args)
        if self.KS_DA_MI_corr_allrep is not None:
            dicttoh5(self.KS_DA_MI_corr_allrep, file_name_root + '_DA_MI_corr_allrep.h5', create_dataset_args=ds_args)
        if self.KS_cov_allrep is not None:
            dicttoh5(self.KS_cov_allrep, file_name_root + '_cov_allrep.h5', create_dataset_args=ds_args)
        if self.eigvec_centrality_da_allrep is not None:
            dicttoh5(self.eigvec_centrality_da_allrep, file_name_root + '_eigvec_centrality_da_allrep.h5',
                     create_dataset_args=ds_args)
        if self.eigvec_centrality_don_allrep is not None:
            dicttoh5(self.eigvec_centrality_don_allrep, file_name_root + '_eigvec_centrality_don_allrep.h5',
                     create_dataset_args=ds_args)
        if self.eigvec_centrality_acc_allrep is not None:
            dicttoh5(self.eigvec_centrality_acc_allrep, file_name_root + '_eigvec_centrality_acc_allrep.h5',
                     create_dataset_args=ds_args)
        if self.eigvec_centrality_da_mean_allrep is not None:
            dicttoh5(self.eigvec_centrality_da_mean_allrep, file_name_root + '_eigvec_centrality_da_mean_allrep.h5',
                     create_dataset_args=ds_args)
        if self.eigvec_centrality_da_indep_sum_allrep is not None:
            dicttoh5(self.eigvec_centrality_da_indep_sum_allrep,
                     file_name_root + '_eigvec_centrality_da_indep_sum_allrep.h5', create_dataset_args=ds_args)

        # Graphs and communities related attributes
        if self.nxGraphs_dict is not None:
            with open(file_name_root + "_nxGraphs.pickle", 'wb') as outfile:
                pickle.dump(self.nxGraphs_dict, outfile)

        if self.predecessors is not None:
            dicttoh5(self.predecessors, file_name_root + "_predecessors.h5", create_dataset_args=ds_args)

        if self.nodes_communities_collect is not None:
            dicttoh5(self.nodes_communities_collect, file_name_root + "_nodesComm.h5", create_dataset_args=ds_args)

        if self.partitions_collect is not None:
            self.partitions_collect = dict(zip(np.arange(len(self.partitions_collect)), self.partitions_collect))
            dicttoh5(self.partitions_collect, file_name_root + "_partitions_collect.h5", create_dataset_args=ds_args)

        if self.distances_collect is not None:
            self.distances_collect = dict(zip(np.arange(len(self.distances_collect)), self.distances_collect))
            dicttoh5(self.distances_collect, file_name_root + "_distances_collect.h5", create_dataset_args=ds_args)

    def load_from_file(self, file_name_root, save_space=False):
        """
        DESCRIPTION
        reads cached data and loads attributes
        """

        # load arrays
        if file_exists(file_name_root + '_cache.hf'):

            with h5py.File(file_name_root + "_cache.hf", "r") as f:
                print("@>: cached file found: loading ", file_name_root + '_cache.hf')
                for key, val in f.items():
                    print("@>:", key, f[key].dtype, f[key].shape, f[key].size)
                    if f[key].size > 1:

                        # stores value in object
                        setattr(self, key, np.zeros(f[key].shape, dtype=f[key].dtype))
                        f[key].read_direct(getattr(self, key))

                    elif f[key].size == 1:

                        # for *scalar* H5Py Dataset, use empty touple.
                        setattr(self, key, f[key][()])
                    print('    Done loading %s attribute' % key)
        else:
            print("@>: file '%s' does not extist" % (file_name_root + "_cache.hf"))

        if file_exists(file_name_root + '_nodes_to_res_dictionary.h5'):
            print('@>: load %s' % file_name_root + '_nodes_to_res_dictionary.h5')
            setattr(self, 'nodes_to_res_dictionary',
                    dictload(file_name_root + '_nodes_to_res_dictionary.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_nodes_dih_to_indices_dictionary.h5'):
            print('@>: load %s' % file_name_root + '_nodes_dih_to_indices_dictionary.h5')
            setattr(self, 'nodes_dih_to_indices_dictionary',
                    dictload(file_name_root + '_nodes_dih_to_indices_dictionary.h5', fmat='hdf5'))

        # coordinates correlation related attributes
        if file_exists(file_name_root + '_nodes_to_res_dictionary.h5'):
            print('@>: load %s' % file_name_root + '_nodes_to_res_dictionary.h5')
            setattr(self, 'nodes_to_res_dictionary',
                    dictload(file_name_root + '_nodes_to_res_dictionary.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_nodes_dih_to_indices_dictionary.h5'):
            print('@>: load %s' % file_name_root + '_nodes_dih_to_indices_dictionary.h5')
            setattr(self, 'nodes_dih_to_indices_dictionary',
                    dictload(file_name_root + '_nodes_dih_to_indices_dictionary.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_coordinates_allreplicas.h5'):
            print('@>: load %s' % file_name_root + '_coordinates_allreplicas.h5')
            setattr(self, 'coordinates_allreplicas',
                    dictload(file_name_root + '_coordinates_allreplicas.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_displacements_allreplicas.h5'):
            print('@>: load %s' % file_name_root + '_displacements_allreplicash5')
            setattr(self, 'displacements_allreplicas',
                    dictload(file_name_root + '_displacements_allreplicas.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_distances_allreplicas.h5'):
            print('@>: load %s' % file_name_root + '_distances_allreplicas.h5')
            setattr(self, 'distances_allreplicas',
                    dictload(file_name_root + '_distances_allreplicas.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_covar_disp_allreplicas.h5'):
            print('@>: load %s' % file_name_root + '_covar_disp_allreplicas.h5')
            setattr(self, 'covar_disp_allreplicas',
                    dictload(file_name_root + '_covar_disp_allreplicas.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_gcc_allreplicas.h5'):
            print('@>: load %s' % file_name_root + '_gcc_allreplicas.h5')
            setattr(self, 'gcc_allreplicas',
                    dictload(file_name_root + '_gcc_allreplicas.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_pcc_allreplicas.h5'):
            print('@>: load %s' % file_name_root + '_pcc_allreplicas.h5')
            setattr(self, 'pcc_allreplicas',
                    dictload(file_name_root + '_pcc_allreplicas.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_dcc_allreplicas.h5'):
            print('@>: load %s' % file_name_root + '_dcc_allreplicas.h5')
            setattr(self, 'dcc_allreplicas',
                    dictload(file_name_root + '_dcc_allreplicas.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_exclusion_matrix_allreplicas.h5'):
            print('@>: load %s' % file_name_root + '_exclusion_matrix_allreplicas.h5')
            setattr(self, 'exclusion_matrix_allreplicas',
                    dictload(file_name_root + '_exclusion_matrix_allreplicas.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_eigenvector_centrality_allreplicas.h5'):
            print('@>: load %s' % file_name_root + '_eigenvector_centrality_allreplicas.h5')
            setattr(self, 'eigenvector_centrality_allreplicas',
                    dictload(file_name_root + '_eigenvector_centrality_allreplicas.h5', fmat='hdf5'))

        # KS analysis attributes
        if not save_space:
            if file_exists(file_name_root + '_bb_distances_allrep.h5'):
                setattr(self, 'bb_distances_allrep',
                        dictload(file_name_root + '_bb_distances_allrep.h5', fmat='hdf5'))
                setattr(self, 'KS_energies_allrep',
                        dictload(file_name_root + '_energies_allrep.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_DA_energies_allrep.h5'):
            print('@>: load %s' % file_name_root + '_DA_energies_allrep.h5')
            setattr(self, 'KS_DA_energies_allrep',
                    dictload(file_name_root + '_DA_energies_allrep.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_DA_MI_corr_allrep.h5'):
            print('@>: load %s' % file_name_root + '_DA_MI_corr_allrep.h5')
            setattr(self, 'KS_DA_MI_corr_allrep',
                    dictload(file_name_root + '_DA_MI_corr_allrep.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_DA_LMI_corr_allrep.h5'):
            print('@>: load %s' % file_name_root + '_DA_LMI_corr_allrep.h5')
            setattr(self, 'KS_DA_LMI_corr_allrep',
                    dictload(file_name_root + '_DA_LMI_corr_allrep.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_cov_allrep.h5'):
            print('@>: load %s' % file_name_root + '_cov_allrep.h5')
            setattr(self, 'KS_cov_allrep',
                    dictload(file_name_root + '_cov_allrep.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_eigvec_centrality_da_allrep.h5'):
            print('@>: load %s' % file_name_root + '_eigvec_centrality_da_allrep.h5')
            setattr(self, 'eigvec_centrality_da_allrep',
                    dictload(file_name_root + '_eigvec_centrality_da_allrep.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_eigvec_centrality_don_allrep.h5'):
            print('@>: load %s' % file_name_root + '_eigvec_centrality_don_allrep.h5')
            setattr(self, 'eigvec_centrality_don_allrep',
                    dictload(file_name_root + '_eigvec_centrality_don_allrep.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_eigvec_centrality_acc_allrep.h5'):
            print('@>: load %s' % file_name_root + '_eigvec_centrality_acc_allrep.h5')
            setattr(self, 'eigvec_centrality_acc_allrep',
                    dictload(file_name_root + '_eigvec_centrality_acc_allrep.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_eigvec_centrality_da_mean_allrep.h5'):
            print('@>: load %s' % file_name_root + '_eigvec_centrality_da_mean_allrep.h5')
            setattr(self, 'eigvec_centrality_da_mean_allrep',
                    dictload(file_name_root + '_eigvec_centrality_da_mean_allrep.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_eigvec_centrality_da_indep_sum_allrep.h5'):
            print('@>: load %s' % file_name_root + '_eigvec_centrality_da_indep_sum_allrep.h5')
            setattr(self, 'eigvec_centrality_da_indep_sum_allrep',
                    dictload(file_name_root + '_eigvec_centrality_da_indep_sum_allrep.h5', fmat='hdf5'))

        # dihedral correlation analysis attributes
        if file_exists(file_name_root + '_dihedrals_allreplicas.h5'):
            print('@>: load %s' % file_name_root + '_dihedrals_allreplicas.h5')
            setattr(self, 'dihedrals_allreplicas',
                    dictload(file_name_root + '_dihedrals_allreplicas.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_disp_from_mean_dih_allreplicas.h5'):
            print('@>: load %s' % file_name_root + '_disp_from_mean_dih_allreplicas.h5')
            setattr(self, 'disp_from_mean_dih_allreplicas',
                    dictload(file_name_root + '_disp_from_mean_dih_allreplicas.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_covar_dih_allreplicas.h5'):
            print('@>: load %s' % file_name_root + '_covar_dih_allreplicas.h5')
            setattr(self, 'covar_dih_allreplicas',
                    dictload(file_name_root + '_covar_dih_allreplicas.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_dih_gcc_allreplicas.h5'):
            print('@>: load %s' % file_name_root + '_dih_gcc_allreplicas.h5')
            setattr(self, 'dih_gcc_allreplicas',
                    dictload(file_name_root + '_dih_gcc_allreplicas.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_dih_pcc_allreplicas.h5'):
            print('@>: load %s' % file_name_root + '_dih_pcc_allreplicas.h5')
            setattr(self, 'dih_pcc_allreplicas',
                    dictload(file_name_root + '_dih_pcc_allreplicas.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_dih_dcc_allreplicas.h5'):
            print('@>: load %s' % file_name_root + '_dih_dcc_allreplicas.h5')
            setattr(self, 'dih_dcc_allreplicas',
                    dictload(file_name_root + '_dih_dcc_allreplicas.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_dih_eigen_centrality_allreplicas.h5'):
            print('@>: load %s' % file_name_root + '_dih_eigen_centrality_allreplicas.h5')
            setattr(self, 'dih_eigen_centrality_allreplicas',
                    dictload(file_name_root + '_dih_eigen_centrality_allreplicas.h5', fmat='hdf5'))

        # Graphs and communities related attributes

        if file_exists(file_name_root + '_predecessors.h5'):
            print('@>: load %s' % file_name_root + '_predecessors.h5')
            setattr(self, 'predecessors',
                    dictload(file_name_root + '_predecessors.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_nodesComm.h5'):
            print('@>: load %s' % file_name_root + '_nodesComm.h5')
            setattr(self, 'nodes_communities_collect',
                    dictload(file_name_root + '_nodesComm.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_partitions_collect.h5'):
            print('@>: load %s' % file_name_root + '_partitions_collect.h5')
            setattr(self, 'partitions_collect',
                    dictload(file_name_root + '_partitions_collect.h5', fmat='hdf5'))

        if file_exists(file_name_root + '_distances_collect.h5'):
            print('@>: load %s' % file_name_root + '_distances_collect.h5')
            setattr(self, 'distances_collect',
                    dictload(file_name_root + '_distances_collect.h5', fmat='hdf5'))

        if file_exists(file_name_root + "_nxGraphs_dict.pickle"):
            print('@>: load %s' % file_name_root + '_nxGraphs_dict.pickle')
            with open(file_name_root + "_nxGraphs_dict.pickle", 'rb') as infile:
                self.nxGraphs_dict = pickle.load(infile)

        # load other attributes
        try:
            #if self.nodes_to_res_dictionary is not None:
            self.nresidues = len(np.unique(np.asarray(list(self.nodes_to_res_dictionary.values()))))
        except AttributeError:
            self.nresidues = np.asarray([len(v) for k, v in
                                         self.nodes_communities_collect['0']['comm_nodes'].items()]).sum()

        if self.nnodes is not None:
            self.nnodes = len(np.unique(np.asarray(list(self.nodes_to_res_dictionary.values()))))

        try:
            #if self.eigvec_centrality_don_allrep is not None:
            self.num_replicas = len(self.eigvec_centrality_don_allrep)
        except TypeError:
            try:
                self.num_replicas = len(self.eigenvector_centrality_allreplicas)
            except TypeError:
                try:
                    self.num_replicas = len(self.dih_eigen_centrality_allreplicas)
                except TypeError:
                    nreplicas = len(self.nodes_communities_collect)
