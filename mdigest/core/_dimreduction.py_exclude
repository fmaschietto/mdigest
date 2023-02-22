"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: fmaschietto, bcallen95"""

from   mdigest.core.imports import *
import mdigest.core.auxiliary as aux

fn_weighted_avg = lambda x: x / x.sum()


class DimReduct:
    """Dimensionality reduction class"""
    def __init__(self):
        """
        Description
        -----------
        Dimensionality reduction

        Attributes
        ----------
        self.topology_files: dict,
            topology files
        self.trajectory_files: dict,
            trajectory files
        self.featselector_dict: dict,
            features selector dictionary with keys specifying the featurizer funcion to pass to pyemma.featurizer and
            values specifying the atom selection string to pass to the featurizer object
        self.selected_features: obj,
            output of featurizer
        self.data: np.ndarray,
            data array, shape(nsamples, nfeatures)
        self.eigenvalues: np.ndarray,
            eigenvalues array
        self.eigenvectors: dict,
            eigenvectors array
        self.pypca: obj
            PCA object
        self.pytica: obj,
            tICA object
        self.skpca: dict,
            sklear_PCA object
        self.fitted_transformed: np.ndarray,
            transformed data
        self.projected_data: np.ndarray,
            projected data
        self.proj_universe: mda.Universe object,
            projected trjajectory loaded into MDAnalysis universe

        Methods
        -------

        """
        self.topology_files = {}
        self.trajectory_files = {}
        self.featselector_dict = {}

        self.selected_features = None
        self.data = None
        self.eigenvalues = None
        self.eigenvectors = None

        self.pypca = None
        self.pytica = None
        self.skpca = None
        self.fitted_transformed = None
        self.projected_data = None
        self.proj_universe = None

    def load_data(self, load_from='from_dictionaries', align=False, z_score=True, **kwargs):
        """
        Load data (in form of a np.ndarray of shape (nsamples, nfeatures))
        options:

        Parameters
        ----------
        load_from: str,
            - 'from_dictionaries' - load the data from a dictionary specifying the trajectories to parse:
                    1) ``topology_dict = {'name_1': '/path/to/topology.psf'}``
                    2) ``trajectory_dict = {'name_1': ['/path/to/trajectory.dcd']}``

            - 'from_data' - provide the data directly (featurized data) as a dictionary with key: 'data' and value: np.ndarray (n_samples, n_features)
        align: bool
            if True align trajectory w.r.t backbone positions frame 0 of the trajectory.
            Specify reference parameter to use a second trajectory as a reference.
        z_score: bool,
            if True apply z_scoring to remove mean from data
        kwargs: dict,
            specify optional parameters for alignment such as the reference trajectory to use for alingment
                - reference: mda.Universe object
        """

        data = []
        if load_from == 'from_data':
            self.data = kwargs['data']
        elif load_from == 'from_dictionaries':
            print('@>: loading data from dictionaries')
            if align:
                print('@>: aligning trajectory...')
                for k, v in self.topology_files.items():
                    traj = md.load(self.trajectory_files[k], top=self.topology_files[k])
                    if kwargs.__contains__('reference'):
                        ref = kwargs['reference']
                        print('@>: using frame 0 of reference %s...' % ref)
                    else:
                        print('@>: using frame 0 of current trajectory as reference')
                        ref = traj
                    traj.superpose(ref, frame=0, atom_indices=traj.topology.select('backbone'),ref_atom_indices=traj.topology.select('backbone')-1, parallel=True)

                    tempdata = traj.xyz[:, traj.topology.select(self.featselector_dict[k]['select']), :]
                    data.append(tempdata)
                data = np.asarray(data, dtype=object)
                data_ = np.concatenate(data, axis=0)
                data_ = np.asarray(data_ * 10, dtype=float)
                disp = aux._compute_displacements(data_)
                self.data = disp

            else:
                data = []
                for k, v in self.topology_files.items():
                    if self.topology_files[k] is None:
                        raise ValueError('call `set_parameter_dictionary()` to fill topology and trajectory information')
                    else:
                        feat = pyemma.coordinates.featurizer(v)
                        f = list(self.featselector_dict[k].keys())[0]
                        print('@>: featurizer {}'.format(f))
                        featfunction = getattr(pyemma.coordinates.featurizer(v), f)
                        print('@>: featfunction is {}'.format(featfunction))

                        if 'select' in f or '' not in list(self.featselector_dict[k].values()):
                            self.selected_features = featfunction(self.featselector_dict[k][f])
                            feat.add_selection(self.selected_features)
                        else:
                            featfunction()
                    tempdata = pyemma.coordinates.load([self.trajectory_files[k]], features=feat)
                    data.append(tempdata)
                data = np.asarray(data, dtype=object)
                data_ = np.concatenate(data, axis=0)
                self.data = data_ * 10
        else:
            raise ValueError('data option not recognized, use either `from_dictionaries` or `from_data`')

        if z_score:
            print('@>: apply z_score to data')
            funz_z_score = lambda x: (x - x.mean() / x.std())
            self.data = funz_z_score(self.data)
        else:
            print('@>: assuming data is already mean free')
        self.data = np.asarray(self.data, dtype=float)


    def set_trajectories(self, trajectories_dict):
        """
        Sets trajectories for which the dimensionality reduction will be performed. The trajectories_dict keys should be the same as in the topologies_dict (see ``set_topologies()``).


        Parameters
        ----------
        trajectories_dict: dict,
            example: ``traj_dict = {'name_1': ['/path/to/traj_1.dcd'], 'name_2': [/path/to/traj_2.dcd']}``

        """
        self.trajectory_files = trajectories_dict


    def set_topologies(self, topologies_dict):
        """
        Sets topology files. The topologies_dict keys should be the same as for the trajectories_dict (see ``set_trajectories()``).

        Parameters
        ----------
        topologies_dict: dict,
            example: ``traj_dict = {'name_1': '/path/to/top_1.psf', 'name_2': /path/to/top_2.psf'}``

        """
        self.topology_files = topologies_dict


    def set_featurizer(self, features_dict):
        """
        Sets the featurizer function which will be used to extract the features from the trajectories. Accepts the following featurizers from pyemma:

        Parameters
        ----------
        features_dict: dict,
            examples:
                1) ``features_dict = {'name_1': {'select': 'protein and name C CA N'}, 'name_2': {'select': 'protein and name C CA N'}}``
                2) ``features_dict = {'name_1': {'add_backbone_torsions': {'selstr':'protein and name C CA N'}}, 'name_2': {'add_backbone_torsions': {'selstr':'protein and name C CA N'}}}``
        """

        self.featselector_dict = features_dict


    def set_parameter_dictionary(self, trajectory_dict, topology_dict, feature_selector_dictionary):
        """
        Collect topology, trajectory and atom group selection information from corresponding dictionaries
        each dictionary should have the same keys and the same number of values as spectified in the documentation
        of each function.

        Parameters
        ----------
        trajectory_dict: dict,
            see ``set_trajectories()``
        topology_dict: dict,
            see ``set_topologies()``
        feature_selector_dictionary: dict,
            see ``set_featurizer()``
        """

        self.set_trajectories(trajectory_dict)
        self.set_topologies(topology_dict)
        self.set_featurizer(feature_selector_dictionary)


    def dimReduction(self, method='PCA', n_components=10, projection=False, **kwargs):
        """
        Performs dimensionality reduction on the data.

        Parameters
        ----------
        method: 'str',
            - 'PCA': use pyemma.coordinates.pca
            - 'tICA': use pyemma.coordinates.tica
            - 'sklearn_PCA': sklearn.decomposition.PCA

        n_components: int (default=10),
            the number of dimensions (independent components) to project onto. -1 means all numerically available dimensions will be used.

        projection: bool,
            whether to save dcd file of the trajectory projected onto selected components. Option available only when using 'PCA' or 'tICA' methods.

        kwargs:
            - whiten: bool,
                whether to whiten data
            - project_on: int, or range
                which eigenvector to use for projection ``kwargs = {'project_on': range(0,2)}`` selects components 0,1.

            - lag: int,
                lagtime for tICA

        """

        if method == 'PCA':
            extra = {}
            print('@>: compute pca')
            if kwargs.__contains__('whiten'):
                print('@>: whiten data = True')
                extra = {'whiten': True}

            self.pypca = pyemma.coordinates.pca(self.data, dim=n_components, **extra)
            print('@>: fit transform')
            self.fitted_transformed = self.pypca.fit_transform(self.data)
            self.eigenvectors = self.pypca.eigenvectors
            self.eigenvalues=self.pypca.eigenvalues
            if projection:
                print('@>: project trajectory onto selected components')
                if kwargs.__contains__('project_on'):
                    if not isinstance(kwargs['project_on'], int):
                        print('@>: components {} are selected for projection'.format(list(kwargs['project_on'])))
                        selected_pc = kwargs['project_on']
                        print('@>: store selected eigenvectors')
                        pc = np.sum(self.pypca.eigenvectors[:, selected_pc], axis=1)
                        print('@>: store transformed coord')
                        fitted_pc = np.sum(self.fitted_transformed[:, selected_pc], axis=1)
                    else:
                        print('@>: selected component 0 for projection')
                        selected_pc = kwargs['project_on']
                        print('@>: store selected eigenvectors')
                        pc = self.pypca.eigenvectors[:, selected_pc]
                        print('@>: store transformed coord')
                        fitted_pc = self.fitted_transformed[:, selected_pc]
                else:
                    print('@>: selected component 0 for projection')
                    selected_pc = 0
                    print('@>: store selected eigenvectors')
                    pc = self.pypca.eigenvectors[:, selected_pc]
                    print('@>: store transformed coord')
                    fitted_pc = self.fitted_transformed[:, selected_pc]

                print('@>: fitted transformed coordinates have shape {}'.format(self.fitted_transformed.shape))
                print('@>: project transformed coordinates')
                projected_data = np.outer(fitted_pc, pc) + self.data.mean(axis=0)
                self.projected_data = projected_data.reshape((len(fitted_pc), -1, 3))
                print('@>: total variance described by selected components',
                      np.sum(fn_weighted_avg(self.pypca.eigenvalues)[selected_pc]))

        elif method == 'tICA':
            print('@>: compute tica')
            if kwargs.__contains__('lag'):
                lag =kwargs['lag']
            else:
                lag=10
            self.pytica = pyemma.coordinates.tica(self.data, lag=lag, dim=n_components)  # **kwargs
            print('@>: fit transform')
            self.fitted_transformed = self.pytica.fit_transform(self.data)

            if projection:
                print('@>: project trajectory onto selected components')
                if kwargs.__contains__('project_on'):
                    print('@>: components {} are selected for projection'.format(list(kwargs['project_on'])))
                    selected_pc = kwargs['project_on']
                else:
                    print('@>: selected component 0 for projection')
                    selected_pc = 0
                print('@>: store selected eigenvectors')
                pc = self.pytica.eigenvectors[:, selected_pc]
                print('@>: store transformed coord')
                fitted_pc = self.fitted_transformed[:, selected_pc]
                print('@>: project transformed coordinates')
                projected_data = np.outer(fitted_pc, pc) + self.data.mean(axis=0)
                self.projected_data = projected_data.reshape((len(fitted_pc), -1, 3))

                print('@>: total variance described by selected components',
                      np.sum(fn_weighted_avg(self.pytica.eigenvalues)[selected_pc]))

        elif method == 'sklearn_PCA':
            # avoid projection when using sklearn, eigensolver uses selects min(n_samples, n_features),
            # such that  the projection (as computed for the `PCA` and `TICA` methods fails)
            # use for plotting PCA SPACE
            self.skpca = sklearn.decomposition.PCA(n_components=n_components)  # , **kwargs
            self.eigenvalues = self.skpca.singular_values_
            self.eigenvectors = self.skpca.components_
            self.fitted_transformed = self.skpca.fit_transform(self.data)
            if kwargs.__contains__('plot_params'):
                labels = kwargs['plot_params']['labels']



        else:
            raise ValueError('method not recognized, use either `PCA`, `tICA`, `sklearn_PCA`')

    def write_proj_trajectory(self, key, **kwargs):
        """
        Writes the projected universe to a file, parameters are passed to the function through **kwargs, as follows:

        Parameters
        ----------
        key: str,
            which trajectory to project, choose one key from specified topologies_dict

        kwargs: dict,
            example: ``kwargs = {'universe' : MDAanalysis.universe,
                                 'selection': 'protein and name C CA N',
                                 'outdir'   : './',
                                 'outname'  : 'projected_traj'}``
        """

        if kwargs.__contains__('universe'):
            universe = kwargs['universe']
            proj_universe = universe

        else:
            topo = None
            traj = None
            if kwargs.__contains__('topology_file'):
                topo = kwargs['topology_file']

            if kwargs.__contains__('trajectory_file'):
                traj = kwargs['trajectory_file']

            else:
                traj = self.trajectory_files[key]

            if kwargs.__contains__('selection'):
                selection = kwargs['selection']
            else:
                selection = 'protein and name C CA N'

            universe = mda.Universe(topo, traj)
            print(universe.trajectory[0].positions)
            proj_universe = mda.Merge(universe.select_atoms(selection))
            print(self.projected_data)
            proj_universe.load_new(self.projected_data, order="fac")

        if kwargs.__contains__('outdir'):
            outdir = kwargs['outdir']
        else:
            outdir = './'

        if kwargs.__contains__('outname'):
            outname = kwargs['outname']
        else:
            outname = 'projected_traj'

        if kwargs.__contains__('step'):
            step = kwargs['step']
        else:
            step = 1

        self.proj_universe = proj_universe

        with mda.Writer(outdir + outname + ".dcd", proj_universe.atoms.n_atoms) as W:
            for ts in proj_universe.trajectory[0::step]:
                W.write(proj_universe.atoms)
        W.close()
        proj_universe.trajectory[0]
        if kwargs.__contains__('savePDB'):
            with mda.Writer(outdir + outname + "_traj.pdb", proj_universe.atoms.n_atoms) as W:
                for ts in proj_universe.trajectory[0::step]:
                    W.write(proj_universe.atoms)
        W.close()
        proj_universe.trajectory[0]
        with mda.Writer(outdir + outname + ".pdb", proj_universe.atoms.n_atoms) as W:
            W.write(proj_universe.atoms)
        W.close()



