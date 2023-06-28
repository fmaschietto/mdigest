# create universe class
from mdigest.utils.imports import *
from mdigest.utils import toolkit as tk

from MDAnalysis.analysis import align as mdaAlign
from MDAnalysis.analysis import rms
import mdtraj as md

import pickle as pkl
import os
from numba import jit, prange
from tqdm.notebook import trange


# global functions
def pkl_dump(todump, filename):
    with open(filename, 'wb') as handle:
        pkl.dump(todump, handle, protocol=pkl.HIGHEST_PROTOCOL)
    handle.close()
    return


class MDAUniverse:
    """General class to handle and align the MDA Universe
    """
    def __init__(self):
        self.traj = None
        self.topo = None

        self.traj_ref = None
        self.topo_ref = None

        self.mda_u = None
        self.mda_ref_u = None

        self.selection_rmsd = None           # indices of atoms to use for rmsd calculation
        self.selection = None                # select indeces of a subset of the topology of interest
        self.selection_alignment = None      # the indices of the atoms to superpose (mobile)
        self.selection_alignment_ref = None  # the indices of the atoms to superpose (reference)

        self.rmsd = None
        self.name = None
        self.box = None
        self.translate = None


    def set_name(self, name):
        self.name = name

    def load_universe(self, topo, traj, topo_ref, traj_ref):
        self.mda_u = mda.Universe(topo, traj)
        self.mda_ref_u = mda.Universe(topo_ref, traj_ref)

    def source_universe(self, universe, reference=None):
        self.mda_u = universe.mda_u
        if reference is not None:
            self.mda_ref_u = reference
        else:
            self.mda_ref_u = universe.mda_u

    def set_selection(self, atm_selection, segIDs):
        """
        Parameters
        ----------
        atm_selection: str
        Atom selection using MDAnalysis selection string

        segIDs: str
        which segids to select

        Example selection : 'not resid 455'
        Example segIDs : '*'
        """
        self.selection_alignment = "segid " + " ".join(segIDs) + " and not (name H* or name [123]H*) and " + atm_selection

    def set_selection_ref(self, atm_selection, segIDs):
        """ Example selection : 'not resid 455'
            Example segIDs : '*'
        """
        self.selection_alignment_ref = "segid " + " ".join(segIDs) + " and not (name H* or name [123]H*) and " + atm_selection

    def set_selection_rmsd(self, selection):
        self.selection_rmsd = selection


    def align_traj(self, inmem=True, aligntoavg=False, strict=True, center=False):
        """
        Use MDAnalysis to align trajectory with respect to a reference

        This function assumes that atoms in reference traj and mobile traj are in the same order.
        if not use align_mdtraj() which allows to select different atoms in mobile and reference.

        Parameters
        ----------
        inmem: bool,
        aligntoavg: bool,
        strict: bool,
        center: bool,

        Returns
        -------
        """

        mda_u = self.mda_u
        mda_ref_u = self.mda_ref_u

        # Set the first frame as reference for alignment
        self.mda_u.trajectory[0]
        self.mda_ref_u.trajectory[0]

        if aligntoavg:
            # Set the average coordinates as reference for alignment
            atomgroup_ref = self.mda_ref_u.select_atoms(self.selection_alignment_ref)
            reference_coordinates = self.mda_ref_u.trajectory.timeseries(asel=atomgroup_ref).mean(axis=1)
            reference = mda.Merge(atomgroup_ref).load_new(reference_coordinates[None, :, :], order="fac")
            mdaAlign.AlignTraj(mda_u, reference, select=self.selection_alignment, in_memory=inmem, weights="mass",
                               strict=strict).run()
        else:
            mdaAlign.AlignTraj(mda_u, mda_ref_u, self.selection_alignment,
                                       verbose=True, in_memory=inMemory, weights="mass", strict=strict).run()

        if center:
            if self.mda_u.trajectory.ts.triclinic_dimensions is not None:
                for ts in self.mda_u.trajectory:
                    protein_center = self.mda_u.select_atoms('protein').center_of_mass(pbc=True)
                    dim = ts.triclinic_dimensions
                    box_center = np.sum(dim, axis=0) / 2
                    self.mda_u.atoms.translate(box_center - protein_center)




    def compute_rmsd(self, superposition=False):
        self.mda_u.trajectory[-1]  # set mobile trajectory to last frame
        self.mda_ref_u.trajectory[0]  # set reference trajectory to first frame
        mobile_ca = self.mda_u.select_atoms(self.selection_rmsd)
        ref_ca = self.mda_ref_u.select_atoms(self.selection_rmsd)
        self.rmsd = rms.rmsd(mobile_ca.positions, ref_ca.positions, superposition=superposition)

    def find_shift(self, box_size=None, offset=0):
        """
        Find how much to shift each coordinate to recenter the
            trajectory in the centrer of the box
        """

        # compute center of mass for each frame of the trajectory
        com = np.zeros((self.mda_u.trajectory.n_frames, 3))
        coords = np.zeros((self.mda_u.trajectory.n_frames, self.mda_u.select_atoms('protein').atoms.n_atoms, 3))
        count = 0
        for ts in self.mda_u.trajectory:
            com[count] = np.asarray(self.mda_u.select_atoms('protein').center_of_mass())
            coords[count] = (np.asarray(self.mda_u.select_atoms('protein').center_of_mass()))
            count += 1

        com = np.asarray(com)
        # average com
        avg_com = com.mean(axis=0)

        # round to nearest multiple of 5
        avg_com = [5 * round(c/5) for c in avg_com]
        print('@>: avg. center of mass:', avg_com)

        # find box dimensions
        if box_size is None:
            minx = np.min(coords[:, :, 0]-offset)
            miny = np.min(coords[:, :, 1]-offset)
            minz = np.min(coords[:, :, 2]-offset)

            maxx = np.max(coords[:, :, 0]+offset)
            maxy = np.max(coords[:, :, 1]+offset)
            maxz = np.max(coords[:, :, 2]+offset)

            a = np.array([minx, miny, minz])
            b = np.array([maxx, maxy, maxz])
            # print('@>: box dimensions:', b-a)
            lenx = np.linalg.norm(b[0] - a[0])
            leny = np.linalg.norm(b[1] - a[1])
            lenz = np.linalg.norm(b[2] - a[2])

            # round box length to nearest multiple of 5
            box_lenx = float(5 * np.round(lenx/5))
            box_leny = float(5 * np.round(leny/5))
            box_lenz = float(5 * np.round(lenz/5))

            self.box = [box_lenx, box_leny, box_lenz]
        else:
            self.box = box_size

        print('@>: box dimensions:', self.box)

        # center of box
        center_of_box = [self.box[0]/2, self.box[1]/2, self.box[2]/2]
        shift_x = -np.round(avg_com[0] - center_of_box[0])
        shift_y = -np.round(avg_com[1] - center_of_box[1])
        shift_z = -np.round(avg_com[2] - center_of_box[2])
        self.translate = [shift_x, shift_y, shift_z]


class MDTUniverse:
    def __init__(self):

        self.traj = None
        self.topo = None

        self.traj_ref = None
        self.topo_ref = None

        self.mdt_u = None
        self.mdt_ref_u = None
        self.selection = None               # select indeces of a subset of the topology of interest
        self.selection_alignment = None     # the indices of the atoms to superpose (mobile)
        self.selection_alignment_ref = None # the indices of the atoms to superpose (reference)
        self.use_frame = 0                  # the index of the conformation in reference to align to, if set_use_frame is not called, defalult is zero
        self.rmsd = None
        self.name = None

        self.segid_map = None
        self.chainid_map = None
        self.aligned = None
        self.dihedrals = None

        self.distances_cache = None
        self.distances = None

        self.box = None
        self.translate = None
        self.log = None


    def set_logfile(self, log):
        self.log = log
        logFile = open(self.log, 'w+')
        logFile.close()

    def set_name(self, name):
        self.name = name

    def set_use_frame(self, frame):
        self.use_frame = frame

    def load_universe(self, topo, traj, topo_ref, traj_ref):
        self.mdt_u = md.load(traj, top=topo)
        self.mdt_ref_u = md.load(traj_ref, top=topo_ref)

        print('@>: trajectory contains {} atoms: '.format(self.mdt_u.n_atoms))
        print('@>: reduce trajectory according to following selection: ', self.selection)
        if self.selection != 'all':
            try:
                self.mdt_u.topology.select(self.selection)
                self.mdt_u, self.mdt_ref_u  = tk.reduce_trajectory()  # Reduce trajectory to selected atoms
            except:
                logFile = open(self.log, 'w+')
                tk.print_screen_logfile("@>: selection string not valid -- default ('all') will be used'", logFile)
                self.selection = 'all'

    def source_universe(self, traj, reference=None):
        if reference is not None:
            self.mdt_ref_u = reference
        else:
            self.mdt_u = traj

    def set_selection(self, group_selection):
        self.selection = group_selection

    def set_selection_alignment_ref(self, atm_selection):
        self.selection_alignment_ref = atm_selection

    def set_selection_alignment(self, atm_selection):
        self.selection_alignment = atm_selection

    def set_mapping_segid(self, mapping):
        # Example mapping=lambda X: 0 if X <= 253 else 1
        self.segid_map = mapping

    def set_mapping_chainid(self, mapping):
        # example mapping={0: 'A', 1: 'B'}
        self.chainid_map = mapping

    def set_distances_cache(self, dist_cache):
        self.distances_cache = dist_cache

    def align_md_traj(self):
        """
        Align MDTraj Trajectory

        Parameters
        ----------

        self.mdt_u: mobile mdtraj universe
        self.mdt_ref_u: reference mdtraj universe
        self.selection_string: selection string for alignment

        :return self.aligned
        """

        traj = self.mdt_u
        traj_ref = self.mdt_ref_u
        selection = self.selection_alignment
        selection_ref = self.selection_alignment

        print('@>: selection for alignment: ', self.selection_alignment)

        atom_indices = traj.topology.select(selection)
        atom_indices_ref = traj_ref.topology.select(selection_ref)

        print('@>: use frame {} for alignment'.format(self.use_frame))

        self.aligned = traj.superpose(traj_ref, frame=self.use_frame, atom_indices=atom_indices, ref_atom_indices=atom_indices_ref,
                                    parallel=True)
        self.mdt_u.Trajectory = self.aligned # in memory update trajectory with aligned version

    def adjust_topology(self):
        aligned = self.aligned.atom_slice(self.aligned.topology.select(self.selection))
        table, bonds = aligned.topology.to_dataframe()
        if self.chainid_map is None:
            pass
        else:
            table['chainID'] = table['resSeq'].map(self.chainid_map)

        if self.segid_map is None:
            pass
        else:
            table['segmentID'] = table['chainID'].map(self.segid_map)

        aligned.topology = md.Topology.from_dataframe(table, bonds)
        self.aligned = aligned

    def compute_angles(self):
        traj = self.aligned
        phi_idces, phi = md.compute_phi(traj, periodic=False, opt=True)
        psi_idces, psi = md.compute_psi(traj, periodic=False, opt=True)
        stack = np.hstack([phi, psi])
        self.dihedrals = stack

    def compute_dihed(self):
        if self.aligned is None:
            print('@>: using non-aligned trajectory')
            traj = self.mdt_u
        else:
            traj = self.aligned
        indices, dihedrals = md.compute_phi(traj)
        indices_psi, dihedrals_psi = md.compute_psi(traj)
        sines = np.sin(dihedrals)
        cosines = np.cos(dihedrals)
        sines_psi = np.sin(dihedrals_psi)
        cosines_psi = np.cos(dihedrals_psi)
        q = np.zeros((sines.shape[0], sines.shape[1] * 4))
        q[:, ::4] = sines
        q[:, 1::4] = cosines
        q[:, 2::4] = sines_psi
        q[:, 3::4] = cosines_psi
        self.dihedrals = q

    def compute_distances(self):
        if self.aligned is None:
            print('@>: using non-aligned trajectory')
            traj = self.mdt_u

        else:
            traj = self.aligned
        cached_distances = os.path.exists(self.distances_cache)
        if not cached_distances:
            distances = np.empty((traj.n_frames, traj.n_frames), dtype=np.float32)
            for i in trange(traj.n_frames):
                distances[i] = md.rmsd(traj, traj, i)
            self.distances = distances

            pkl_dump(distances, self.distances_cache)
        else:
            with open(self.distances_cache, 'rb') as handle:
                unserialized_data = pkl.load(handle)
                distances = unserialized_data
                handle.close()
            print('@>: load distances from cache')
            self.distances = distances
        return distances

    @jit(nopython=True, parallel=False, cache=True, nogil=True)
    def calc_rmsd_2frames(self, ref, frame):
        """
        RMSD calculation between a reference and a frame.
        This function is "jitted" for better performances
        """
        dist = np.zeros(len(frame), dtype=np.float32)
        for atom in range(len(frame)):
            dist[atom] = ((ref[atom][0] - frame[atom][0]) ** 2 +
                          (ref[atom][1] - frame[atom][1]) ** 2 +
                          (ref[atom][2] - frame[atom][2]) ** 2)
        return np.sqrt(dist.mean())

    def find_shift(self, box_size=None, offset=0):
        """
        Find how much to shift each coordinate to recenter the
        trajectory in the centrer of the box
        """

        if self.aligned is None:
            print('@>: using non-aligned trajectory')
            trajectory = self.mdt_u
        else:
            trajectory = self.aligned
        # compute center of mass for each frame of the trajectory
        com = md.compute_center_of_mass(trajectory)
        # average com
        avg_com = com.mean(axis=0)*10

        # round to nearest multiple of 5
        avg_com = [5 * round(c/5) for c in avg_com]
        print('@>: avg. center of mass:', avg_com)

        # find box dimensions
        if box_size is None:
            minx = np.min(trajectory.xyz[:, :, 0])*10-offset
            miny = np.min(trajectory.xyz[:, :, 1])*10-offset
            minz = np.min(trajectory.xyz[:, :, 2])*10-offset

            maxx = np.max(trajectory.xyz[:, :, 0])*10+offset
            maxy = np.max(trajectory.xyz[:, :, 1])*10+offset
            maxz = np.max(trajectory.xyz[:, :, 2])*10+offset

            a = np.array([minx, miny, minz])
            b = np.array([maxx, maxy, maxz])
            lenx = np.linalg.norm(a[0] - b[0])
            leny = np.linalg.norm(a[1] - b[1])
            lenz = np.linalg.norm(a[2] - b[2])

            # round box length to nearest multiple of 5
            box_lenx = float(5 * np.round(lenx/5))
            box_leny = float(5 * np.round(leny/5))
            box_lenz = float(5 * np.round(lenz/5))

            self.box = [box_lenx, box_leny, box_lenz]
        else:
            self.box = box_size

        print('@>: box dimensions:', self.box)

        # center of box
        center_of_box = [self.box[0]/2, self.box[1]/2, self.box[2]/2]
        shift_x = -np.round(avg_com[0] - center_of_box[0])
        shift_y = -np.round(avg_com[1] - center_of_box[1])
        shift_z = -np.round(avg_com[2] - center_of_box[2])
        self.translate = [shift_x, shift_y, shift_z]

    def centroid(self):
        """
        Compute the centroid of a distance matrix
        """
        std_dev = np.nanstd(self.distances)
        beta = 1
        cluster_centroid = np.exp(-beta * self.distances / std_dev).sum(axis=1).argmax()
        return cluster_centroid