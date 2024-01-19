import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align as mda_align
from sklearn.decomposition import PCA

class DimensionalityReduction:

    def __init__(self):

        self.mda_u = None
        self.projected_u = None
        self.atomgroup = None
        self.reference_u = None
        self.selection = None
        self.data = None
        self.data2D = None
        self.transformed = None
        self.projected = None
        self.variance = None
        self.p_components = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.mean = None
        self.natoms = None
        self.nframes = None
        self.spatial_dim = None
        self.explained_variance_ratio = None
        self._dimensionality_reduction = False


    def source_system(self, align=True, **kwargs):
        self.mda_u = kwargs.get('universe')
        self.reference_u = kwargs.get('reference')
        self.selection = kwargs.get('selection')


        if kwargs.__contains__('start'):
            start = kwargs.get('start')
            start = int(start)
        else:
            start = 0
        if kwargs.__contains__('stop'):
            stop = kwargs.get('stop')
            stop = int(stop)
        else:
            stop = -1
        if kwargs.__contains__('step'):
            step = kwargs.get('step')
            step = int(step)
        else:
            step = 1

        if align:
            universe = self.mda_u
            atomgroup = universe.select_atoms(self.selection)
            # prealign to first frame
            prealigner = mda_align.AlignTraj(self.mda_u, self.mda_u, select="backbone", in_memory=True).run()
            # get reference coordinates
            reference_coordinates = self.mda_u.trajectory.timeseries(asel=atomgroup).mean(axis=1)
            self.reference_u = mda.Merge(atomgroup).load_new(reference_coordinates[None, :, :], order="fac")
            aligner = mda_align.AlignTraj(self.mda_u, self.reference_u, select='backbone', in_memory=True).run()

        # Get the indices of the atoms that match the selection.
        atom_group = self.mda_u.select_atoms(self.selection)
        idices_selection = atom_group.indices

        # Get the flattened coordinates of each frame.
        coordinates = []
        for frame in self.mda_u.trajectory:
            coordinates.append(frame.positions[idices_selection].ravel())
        coordinates = np.array(coordinates)

        self.natoms, self.nframes, self.spatial_dim = self.mda_u.trajectory.timeseries(asel=atom_group).shape
        self.data2D = coordinates
        self.data = coordinates.reshape((self.nframes, self.natoms, self.spatial_dim))
        self.mean = np.mean(self.data, axis=0)


    def MVD(self, n_components=None):
        """Find directions of Maximum Variance by applying PCA on the trajectory."""
        pca = PCA()
        pca.fit(self.data2D)
        # eigenvalues and eigenvectors
        self.eigenvectors = pca.components_
        self.eigenvalues = pca.explained_variance_
        self.explained_variance_ratio = pca.explained_variance_ratio_

        # variance and cumulated variance and reduced data
        if n_components is None:
            n = len(self.eigenvalues)
            self.variance = self.eigenvalues
            self.p_components = self.eigenvectors

        else:
            n = n_components
            self.variance = self.eigenvalues[:n]
            self.p_components = self.eigenvectors[:n, :]

        self._dimensionality_reduction = True

        pca.n_components = n

        self.transformed = pca.transform(self.data2D)



    def transformation(self, universe, components=None):
        """
        Transformation projects the data into the new space defined by the principal components.
        """

        if self._dimensionality_reduction is False:
            print('@>: Perform dimensionality reduction first!')
            return

        pcs = self.eigenvectors[:components, :]
        trans = self.transformed[:, :components]
        projected = np.outer(trans, pcs) + self.mean.flatten()

        self.projected = projected.reshape((len(trans), -1, self.spatial_dim))
        self.projected_u = universe.load_new(self.projected, order="fac")


    def to_file(self, **kwargs):
        """
        Writes the projected universe to a file, parameters are passed to the function through **kwargs, as follows:

        Parameters
        ----------
        kwargs: dict,
            example: ``kwargs = {'universe' : MDAanalysis.universe,
                                 'selection': 'protein and name C CA N',
                                 'outdir'   : './',
                                 'outname'  : 'projected_traj'}``
        """

        if kwargs.__contains__('universe'):
            universe = kwargs['universe']

        elif self.mda_u is not None:
            universe = self.mda_u

        else:
            topo = None
            traj = None
            if kwargs.__contains__('topology_file') and kwargs.__contains__('trajectory_file'):
                topo = kwargs['topology_file']
                traj = kwargs['trajectory_file']

            else:
                print('@>: Provide a topology and trajectory keys in parameters dictionary or MDAnalysis Universe')

            if kwargs.__contains__('selection'):
                selection = kwargs['selection']
            else:
                selection = 'protein and name C CA N'

            universe = mda.Universe(topo, traj)

        if kwargs.__contains__('outdir'):
            outdir = kwargs['outdir']
        else:
            outdir = './'

        if kwargs.__contains__('outname'):
            outname = kwargs['outname']
        else:
            outname = 'projected_traj'

        if kwargs.__contains__('start'):
            start = kwargs.get('start')
            start = int(start)
        else:
            start = 0
        if kwargs.__contains__('stop'):
            stop = kwargs.get('stop')
            stop = int(stop)
        else:
            stop = -1
        if kwargs.__contains__('step'):
            step = kwargs.get('step')
            step = int(step)
        else:
            step = 1
        
        self.projected_u.trajectory[0]

        with mda.Writer(outdir + outname + ".xtc", self.projected_u.atoms.n_atoms) as W:
            for ts in self.projected_u.trajectory[start:stop:step]:
                W.write(self.projected_u.atoms)
        W.close()
        self.projected_u.trajectory[0]
        if kwargs.__contains__('savePDB'):
            with mda.Writer(outdir + outname + "_traj.pdb", self.projected_u.atoms.n_atoms) as W:
                for ts in self.projected_u.trajectory[start:stop:step]:
                    W.write(self.projected_u.atoms)
        W.close()
        self.projected_u.trajectory[0]
        with mda.Writer(outdir + outname + ".pdb", self.projected_u.atoms.n_atoms) as W:
            W.write(self.projected_u.atoms)
        W.close()

        return self.projected_u

