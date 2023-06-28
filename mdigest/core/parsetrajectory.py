"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: fmaschietto, bcallen95"""

from mdigest.core.imports                 import *


class MDS:
    """Parse molecular dynamics trajectories"""
    def __init__(self):
        """
        General purpose class to parse molecular dynamics trajectories, based on MDAnalysis.
        ``[**]`` function structure adapted from https://github.com/melomcr/dynetan

        Attributes
        ----------
        self.mds_data: class object,
            for saving
        self.mda_u: object,
            MDAnalysis universe
        self.atom_group_selstr: str,
            atom group selection string
        self.atom_group_selection: object,
            atom group selected based on atom_group_selstr from mda_u.select_atoms(atom_group_selstr)
        self.nodes_idx_array: np.ndarray,
            indices of selected atoms
        self.nodes_to_res_dictionary: dict
            dictionary with nodes (atoms) indices as keys and residues as values
        self.natoms: int,
            number of atoms in selected atom group
        self.total_nframes: int,
            total number of frames i MDA universe before slicing and trimming
        self.nframes_per_replica: int,
            number of frames per replica
        self.num_replicas: int,
            number of replicas
        self.nresidues: int,
            number of residues in selected atom group
        self.initial: int,
            starting frame read in from trajectory
        self.final: int,
            final frame read in from trajectory
        self. step: int,
            step between each frame
        self.window_span: int,
            window span

        References
        ----------

        Examples
        --------
        """
        self.mds_data                = None
        self.mda_u                   = None
        self.atom_group_selstr       = None
        self.system_selstr           = None
        self.atom_group_selection    = None
        self.nodes_idx_array         = None
        self.nodes_to_res_dictionary = None

        self.natoms              = 0
        self.total_nframes       = 0
        self.nframes_per_replica = 0
        self.num_replicas        = 0
        self.nresidues           = 0
        self.nnodes              = 0
        self.initial             = 0
        self.final               = 0
        self.step                = 0
        self.window_span         = 0


    def set_num_replicas(self, num_replicas):
        """
        Set the number of replicas

        Parameters
        ----------
        num_replicas: int,
            number of concatenated replicas
        """

        self.num_replicas = num_replicas


    def source_system(self, mda_universe):
        """
        Source MDA universe

        Parameters
        ----------
        mda_universe: mda.Universe object,
            MDA universe object
        """
        self.mda_u = mda_universe.copy()


    def load_system(self, topology, traj_files, inmem=True):
        """
        Load MDA universe from topology and trajectory

        Parameters
        ----------
        topology: str,
            path to topology file
        traj_files: str or list of str,
            strings or list of strings specifying the path to trajectory file/s
        inmem: bool,
            whether to load trajectory in memory
        """
        self.mda_u = mda.Universe(topology,traj_files, in_memory=inmem)


    def align_traj(self, inmem=True, reference=None, selection='protein'):
        """
        Align trajectory to specified selection using aling protocol from MDAnalysis

        Parameters
        ----------
        inmem: bool, default True,

        reference: bool or None, defalult None,
            a reference universe can be specified to use against for alignment
        selection: str,
            selection string to select atoms against which to perform alignment
        """
        from MDAnalysis.analysis import align as mdaAlign

        # Set the first frame as reference for alignment
        self.mda_u.trajectory[0]

        if reference is None:
            alignment = mdaAlign.AlignTraj(self.mda_u, self.mda_u, select="%s and not (name H* or name [123]H*)" % selection,
                                       verbose=True, in_memory=inmem, weights="mass")
            alignment.run()
        else:
            alignment = mdaAlign.AlignTraj(self.mda_u, reference, select="%s and not (name H* or name [123]H*)" % selection,
                                       verbose=True, in_memory=inmem, weights="mass")
            alignment.run()


    def set_selection(self, atom_group_selstr='protein and name CA', system_selstr='protein'):
        """
        Set selection strings

        Parameters
        ----------
        system_selstr: str,
            selection string to select the system portion to consider when computing the exclusion matrix
            for example "protein"
        atom_group_selstr: str,
            selection string to be used for selecting the subset of the nodes on which to perform analysis
            for example "protein and name CA"
        """
        self.atom_group_selstr = atom_group_selstr
        self.system_selstr = system_selstr


    def get_universe(self):
        """
        Retrieve universe
        """
        return self.mda_u


    def stride_trajectory(self, initial=0, final=-1, step=1):
        """
        Stride trajectory

        Parameters
        ----------
        initial: int,
            initial frame from which to start reading in the trajectory
        final: int,
            final frame to consider when reading in the trajectory
        step: int,
            step to use when slicing the traj frames
        """

        # Select atom group of interest
        self.atom_group_selection = self.mda_u.select_atoms(self.atom_group_selstr)

        # Set natoms attribute as the number of atoms in the selected atom group
        self.natoms  = self.atom_group_selection.atoms.n_atoms

        # Set nodes_idx_array attribute from the array of indices of each atom in the selected atom group
        # Each atom is assigned an index (interger between 0 and N with N total number of atoms in the MDUniverse)
        self.nodes_idx_array = self.atom_group_selection.ix_array

        if final < 0:
            final = self.mda_u.trajectory.n_frames
        else:
            final = final

        # Total number of frames in the original trajectory
        self.total_nframes = len(self.mda_u.trajectory)

        # Number or replicas
        num_replicas = self.num_replicas

        # Initial frame
        self.initial = initial

        # Final frame
        self.final = final

        # Step
        self.step =  step

        # Window span defines the lenght of each simulation block (replica)
        #self.window_span = int(np.ceil(np.ceil((self.final - self.initial) / self.step)/ self.num_replicas))
        self.window_span = ((self.final - self.initial) // self.step) // self.num_replicas

        # Number of frames per replica
        self.nframes_per_replica = int(len(self.mda_u.trajectory[initial:final:step])/self.num_replicas)


        print('@>: number of frames:      %d' % self.total_nframes)
        print('@>: number of replicas:    %d' % self.num_replicas)
        print("@>: using window length of %d simulation steps" % self.window_span)
        print('@>: number or frames per replica: %d' % self.nframes_per_replica)
        print('@>: first frame:           %d' % initial)
        print('@>: last frame:            %d' % final  )
        print('@>: step:                  %d' % step   )

        # Number of residues in selected atom group
        self.nresidues = len(self.atom_group_selection.residues)
        self.nnodes    = len(self.atom_group_selection.resids)
        if self.nnodes != self.nresidues:
            print('@>: warning: selected nodes exceed number of residues')
        print('@>: number of residues in selected atom group: %d' % self.nresidues)
        print('@>: number of nodes    in selected atom group: %d' % self.nnodes)
        print('@>: number of elements in selected atom group: %d' % len(self.atom_group_selection))

        self.nodes_to_res_dictionary = dict(zip(self.mda_u.select_atoms(self.atom_group_selstr).atoms.ids,
                                          self.atom_group_selection.atoms.resindices))

