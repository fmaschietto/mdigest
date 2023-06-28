"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: fmaschietto, bcallen95"""

from mdigest.core.imports import *
import mdigest.core.auxiliary as aux
from MDAnalysis.analysis.dihedrals import Janin
from MDAnalysis.analysis import dihedrals


class MDS_analysis:
    """
    Basic molecular dynamics analysis
    """

    def __init__(self):
        """
        Description
        -----------
        Given a molecular dynamics trajectory computes general anlalyses such as RMSF, RMSD, secondary structure analysis.
        """
        self.mds_data                  = None
        self.mda_u                     = None
        self.mdt_obj                   = None
        self.segIDs                    = None
        self.atm_str_sel               = None
        self.sys_str_sel               = None
        self.atom_group_selection      = None
        self.atom_type_selection       = None
        self.chi1_selection            = None

        # ---------------------------------------------#
        self.chi1_angles                        = None
        self.dihed_angles                       = None
        self.ss_stats                           = None
        self.rgyr_results                       = None
        self.rms_data                           = None
        self.nh_selections                      = None
        self.order_parameters                   = None
        self.pdb_df                             = None
        # ---------------------------------------------#

        self.natoms              = 0
        self.total_nframes       = 0
        self.nframes_per_replica = 0
        self.num_replicas        = 0
        self.nresidues           = 0
        self.initial             = 0
        self.final               = 0
        self.step                = 0
        self.window_span         = 0


    def set_num_replicas(self, num_replicas):
        self.num_replicas = num_replicas


    def load_system(self, topology, traj_files, inmem=True):
        self.mda_u = mda.Universe(topology,traj_files,in_memory=inmem)
        self.natoms = self.mda_u.atoms
        self.mdt_obj = md.load(traj_files, top = topology)
        return


    def align_traj(self, inmem=True, reference=None, selection='protein'):
        from MDAnalysis.analysis import align as mdaAlign
        # Set the first frame as reference for alignment
        self.mda_u.trajectory[0]
        if reference is None:
            alignment = mdaAlign.AlignTraj(self.mda_u, self.mda_u, select="segid " + " ".join(self.segIDs) + " and not (name H* or name [123]H*)",
                                       verbose=True, in_memory=inmem, weights="mass")
            alignment.run()
        else:
            alignment = mdaAlign.AlignTraj(self.mda_u, reference, select=selection,
                                       verbose=True, in_memory=inmem, weights="mass")
            alignment.run()
        return


    def set_node_type_sele(self, node_type_sele):
        self.atom_type_selection = node_type_sele
        return self.atom_type_selection


    def set_selection(self, atm_str_sel, sys_str_sel):
        self.atm_str_sel = atm_str_sel
        self.sys_str_sel = sys_str_sel
        return


    def get_universe(self):
        return self.mda_u


    def do_ss_calculation(self, simple=True):
        """
        Calculation of SS propensities using an MDTraj
        trajectory object. The `simple` option is a Boolean flag that allows for
        the choice of simple SS definitions (Helix, Strand, Coil) or the more
        descriptive SS definitions.

        Parameters
        ----------
        simple: bool
            If true, only 'H', 'E' and 'C' secondary structure assignments are used.

        Returns
        --------
        `self.ss_stats`: pd.DataFrame,
            SS assignments at each frame as well as the SS propensities computed over the course of the trajectory (i.e., % Helix, % Strand, % Coil).
        """
        def _vec_query(arr, my_dict):
            return np.vectorize(my_dict.__getitem__)(arr)

        mdt_u = self.mdt_obj
        sec_struct = md.compute_dssp(mdt_u, simplified=simple)
        if simple:
            ss_defs = {'H': 'Helix', 'E': 'Strand', 'C': 'Coil', 'NA': 'Not Protein'}
        else:
            ss_defs = {'H': 'Alpha Helix', 'B': 'Beta-Bridge',
                      'E': 'Extended Strand/Beta Ladder', 'G': '3/10 Helix',
                      'I': 'Pi Helix', 'T': 'Hydrogen-Bonded Turn',
                      'S': 'Bend', ' ': 'General Coil/Loops/Irregular Elements',
                      'NA': 'Not Protein'}
        ss_labels = _vec_query(sec_struct, ss_defs)
        nframe = sec_struct.shape[0]
        nres = sec_struct.shape[-1]
        resi = [i for i in mdt_u.top.residues]
        resname = [i.name for i in resi]
        resnum = [i.resSeq for i in resi]
        out_df = pd.DataFrame(data=ss_labels.T)
        out_df.insert(0, 'Res. Names', np.asarray(resname))
        out_df.insert(1, 'Res. Num.', np.asarray(resnum))
        out_df['% Helix'] = 0.00
        out_df['% Strand'] = 0.00
        out_df['% Coil'] = 0.00
        for mm in range(nres):
            temp_val = out_df.iloc[mm, 2:].value_counts()
            all_keys = list(temp_val.keys())
            for key in all_keys:
                if key == 'Coil':
                    out_df.loc[mm, '% Coil'] = temp_val[key]/nframe*100.00
                if key == 'Strand':
                    out_df.loc[mm, '% Strand'] = temp_val[key]/nframe*100.00
                if key == 'Helix':
                    out_df.loc[mm, '% Helix'] = temp_val[key]/nframe*100.00
        self.ss_stats = out_df


    def compute_radius_of_gyration(self, **kwargs):
        """
        Calculate the radius of gyration given an MDAnalysis.Universe and a dictionary of atom selections.
        This dictionary should take the following form:

        Parameters
        ----------
        kwargs: dict,
                - selection_id: string to describe the selection
                - selection_str: string used for creating selection with MDA.Universe
        """
        kw_args = kwargs['kwargs']
        temp_rgyr_dict = {}
        for ii, arg in enumerate(kw_args):
            temp_sel = self.mda_u.select_atoms(kw_args[arg])
            temp_list = []
            for ts in self.mda_u.trajectory:
                temp_list.append([self.mda_u.trajectory.time, temp_sel.radius_of_gyration()])
            temp_rgyr_dict[arg] = {'Selection String': kw_args[arg], 'rgyr': np.asarray(temp_list)}

        self.rgyr_results = temp_rgyr_dict
        return


    def calc_chi1(self):
        """
        Calculates the Chi 1 Angles for an MDAnalysis.Universe.
        Uses the following attributes:
        - self.mda_u: The MDAnalysis.Universe
        - self.chi1_selection: a string used to select the subset of the universe for which you want to calculate the Chi 1 Angles.

        Returns
        -------
        self.chi1_angles: an array representing the Chi 1 Angles of your selection at each frame of the trajectory.
        """

        temp_sel = self.mda_u.select_atoms(self.chi1_selection)
        temp_janin = Janin(temp_sel)
        janin_results = temp_janin.run()
        self.chi1_angles = janin_results.results['angles'][:, :, 0]
        return


    def calc_rms_quant(self, sel_str, **kwargs):
        """
        Calculate RMS Quantities (i.e., RMSD, RMSF).

        Parameters
        ----------
        sel_str: str or int,
            string/frame_index describing the reference with which to calculate the RMS quantity
        kwargs: dict
            example:
                ``kwargs={'Initial': 0, 'Average': 'average', 'Final': -1}``
            
        Returns
        -------
        self.rms_data --> A dict with the following structure:
            ``self.rms_data = {'RMSD': {selection_title: RMSD_Value}, 'RMSF': {selection_title: RMSF_Value}}``,
            where RMSD_Value/RMSF_Value are np.arrays containing the computed quantities.
        """

        # Extract the coordinates into array of shape (n_frames, n_atoms, 3)
        coords = aux._extract_coords(self.mda_u, sel_str)
        # Parse the key-word arguments:
        fancy_str = []
        reference_selections = []
        for ii, arg in enumerate(kwargs['kwargs']):
            fancy_str.append(arg)
            reference_selections.append(kwargs['kwargs'][arg])
        # Iterate through the reference_selection list:
        for ii, refsel in enumerate(reference_selections):
            results_dictionary = {}
            ref_coords = None
            if type(refsel) is not type(0):
                if (refsel.lower() == 'average') or (refsel == 'Average'):
                    ref_coords = np.mean(coords, axis=0)
            else:
                ref_coords = coords[refsel]
            n_atoms = len(ref_coords)
            n_frames = len(coords)
            ref_coords = ref_coords.reshape((1, n_atoms, 3))
            deviation = coords - ref_coords
            # Up to here, RMSD and RMSD are the same (both take norm, but different divisor)
            sq_fluct = np.linalg.norm(deviation, axis=-1)
            sq_dev = sq_fluct/n_atoms
            rmsd = np.sqrt(np.sum(sq_dev, axis=-1))
            mean_sq_fluct = np.sum(sq_fluct, axis=0)/n_frames
            rmsf = np.sqrt(mean_sq_fluct)
            results_dictionary['RMSD'] = {fancy_str[ii]: rmsd}
            results_dictionary['RMSF'] = {fancy_str[ii]: rmsf}

            if self.rms_data is None:
                self.rms_data = results_dictionary
            else:
                self.rms_data['RMSD'][fancy_str[ii]] = rmsd
                self.rms_data['RMSF'][fancy_str[ii]] = rmsf


    def do_dihedral_calcs(self, save_data=None):
        """
        Calculate Phi, Psi Angles from input:
        
        Parameters
        ----------
        save_data: dict or None,
            use for saving the output. Requires dict containing the keys:
            - "Directory": The desired directory to save the output to
            - "Output Descriptor": simple string to identify the system uniquely (for saving output)
        """
        str_sel = self.atm_str_sel
        temp_diheds = dihedrals.Ramachandran(self.mda_u.select_atoms(str_sel)).run()
        temp_phi = temp_diheds.angles[:, :, 0]
        temp_psi = temp_diheds.angles[:, :, 1]
        self.dihed_angles = {'Phi': temp_phi, 'Psi': temp_psi}
        if type(save_data) is None:
            pass
        else:
            temp_dir = save_data['Directory']
            if temp_dir.endswith('/'):
                temp_dir = temp_dir[:-1]
            else:
                temp_dir = temp_dir

            np.savez('{}/{}-Dihedrals.npz'.format(temp_dir, save_data['Output Descriptor']), phi=temp_phi, psi=temp_psi)


    def set_NH_selections(self, amide_string_list):
        """
            Expected input:
                - amide_string_list: a list of strings used to make the atom selection for your
                    backbone amide N atoms and the backbone amide H atoms
        """
        self.nh_selections = dict(zip(['N-Atom', 'H-Atom'], amide_string_list))
        return


    def calc_NH_order_params(self):
        """
        This function calculates the S2 amide order parameters according to
        [J. Am. Chem. Soc. 1998, 120, 5301-5311].
        Expected input:
        - set ``self.nh_selections`` before calling this function

        Returns
        -------
        self.order_parameters: pd.DataFrame

        TODO: This function will eventually need some kwargs to specify the N-atom/NH-atom selections.
        """


        def _calc_order_param(nh_vectors):
            """
            Function to actually calculate the order parameters from the
            amide N-H bond vectors.
            """
            oparam = 1.5 * (np.mean(nh_vectors[:, :, 0]**2, axis=0)**2 # <x^2>^2
                            + np.mean(nh_vectors[:, :, 1]**2, axis=0)**2 # <y^2>^2
                            + np.mean(nh_vectors[:, :, 2]**2, axis=0)**2 # <z^2>^2
                            + 2.*(np.mean(nh_vectors[:, :, 0]*nh_vectors[:, :, 1], axis=0)**2) # 2 <xy>^2
                            + 2.*(np.mean(nh_vectors[:, :, 0]*nh_vectors[:, :, 2], axis=0)**2) # 2 <xz>^2
                            + 2.*(np.mean(nh_vectors[:, :, 1]*nh_vectors[:, :, 2], axis=0)**2) # 2 <yz>^2
                            ) - 0.5
            return oparam

        # This function definitely can be improved!!!! As you can see, it currently
        # extracts the coordinates of all atoms and then grabs a subset using selection strings.
        if self.nh_selections is None:
            print('WARNING: You need to set your selection strings! Please call `self.set_NH_selections`')
        else:
            # Create atom selections for the backbone NH atoms
            n_idz = self.mda_u.select_atoms(self.nh_selections['N-Atom']).atoms.indices
            h_idz = self.mda_u.select_atoms(self.nh_selections['H-Atom']).atoms.indices
            if len(n_idz) == len(h_idz):
                pass
            else:
                n_idz = n_idz[1:]
            # Extract the coordinates from the universe.
            temp_xyz = aux._extract_coords(self.mda_u) # shape: (nframes, natoms, 3)
            # Use the atom indices from our atom selections to grab their respective coordinates.
            n_coords = temp_xyz[:, n_idz, :]
            h_coords = temp_xyz[:, h_idz, :]
            # Construct our Relative Vector pointing in the direction of the NH Bond: N ----> H
            nh_vects = h_coords - n_coords
            nh_norm = np.linalg.norm(nh_vects, axis=-1) # Need to make it a unit vector
            nh_vects[:, :, 0] /= nh_norm # Normalize the X-component
            nh_vects[:, :, 1] /= nh_norm # Normalize the Y-component
            nh_vects[:, :, 2] /= nh_norm # Normalize the Z-component
            temp_nh_o2 = _calc_order_param(nh_vects) # calculate S^2 with unit vectors.
            # NH Vector Order Parameter DF
            nh_ops = pd.DataFrame()
            nh_ops['Res. Num.'] = self.mda_u.atoms[n_idz].resnums
            nh_ops['Res. Name'] = self.mda_u.atoms[n_idz].resnames
            nh_ops['NH Order Param.'] = temp_nh_o2
            self.order_parameters = nh_ops
            return


    def stride_trajectory(self, initial=0, final=-1, step=1):
        """
        Stride trajectory

        Parameters
        ----------
        initial: int,
        final: int,
        step: int

        # TODO redundant with MDS, eventually remove and inherit mdigest.MDS object
        """

        # Select atom group of interest
        self.atom_group_selection = self.mda_u.select_atoms(self.atm_str_sel)
        if final < 0:
            final = self.mda_u.trajectory.n_frames
        else:
            final = final
        self.total_nframes = len(self.mda_u.trajectory)
        self.nframes_per_replica = len(self.mda_u.trajectory[initial:final:step])
        # Number or replicas
        num_replicas = self.num_replicas
        self.initial = initial
        self.final = final
        self.step =  step
        self.window_span = int(np.ceil(np.ceil((self.final - self.initial) / self.step)/ num_replicas))
        print('@>: number of frames:      %d' % self.total_nframes)
        print('@>: number of replicas:    %d' % self.num_replicas)
        print("@>: using window length of %d simulation steps" % self.window_span)
        print('@>: first frame:           %d' % initial)
        print('@>: last frame:            %d' % final  )
        print('@>: step:                  %d' % step   )
        self.nresidues = len(self.atom_group_selection.residues)
        print('@>: number of elements in selected atom group %d' % len(self.atom_group_selection))


    def _read_pdb_file(self, filename):
        """
        Utility Function for reading a reference PDB File into a Pandas DF
        For more complex pdb to file operations use mdigest.utils
        """
        out_lists = []
        with open(filename) as file:
            for line in file.readlines():
                if line.startswith('ATOM'):
                    out_lists.append([line[:6].strip(' '), int(line[6:11].strip(' ')),  line[12:16].strip(' '), line[16],
                                      line[17:20], line[21], int(line[22:26].strip(' ')),  line[26],
                                      [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                                      float(line[54:60]), float(line[60:66]), line[72:76].strip(' '),
                                      line[76:78].strip(' '), [i if len(i) > 0 else '0' for i in line[78:80]][0]])
        column_names = ['Record Name', 'Atom Serial No.', 'Atom Name', 'Alt. Loc.', 'Res. Name', 'Chain ID',
                        'Res. No.', 'Ins. Code', 'Coordinates', 'Occ.', 'B Factor', 'Seg. ID', 'Atom Type', 'Charge']
        self.pdb_df = pd.DataFrame(out_lists, columns = column_names)
        return

    def _df_to_pdb(self, desired_outname):
        """
        Minimal function to write quantities like centralities, order_params into B-factor column
        For more complex pdb to file operations use mdigest.utils
        """
        df_vals = self.pdb_df.values
        with open(desired_outname, 'w') as file:
            for line in df_vals:
                strtline, serial, name, altloc, resname, chainid, reseq, icode, pos, occup, bfact, segid, atom_type, charge = line
                file.write('{:<6s}{:5d}  {:<4s}{:<1s}{:<3s}{:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}      {:<4s}{:<2s}{:<2s}\n'.format(strtline, serial, name, altloc, resname, chainid, reseq, icode, pos[0], pos[1], pos[2], occup, bfact, segid, atom_type, charge))
            file.write('TER   \n')
            file.write('END\n')
        return 'Your PDB File is written.'


    ######################## Load Class from cache ######################################
    def load_class_from_file(self, file_name_root):
        def _load_pickle_obj(name):
            with open(str(name)+'.pickle', 'rb') as f:
                return pickle.load(f)

        tmp_path = '{}_sel_str.txt'.format(file_name_root)
        if os.path.exists(tmp_path):
            with open(tmp_path, 'r') as f:
                self.atom_type_selection = ''.join(f.readlines()[0].split('_'))
        else:
            print('Atom Type Selection file does not exist.')

        tmp_chi1_path = '{}_chi1_sel_str.txt'.format(file_name_root)
        if os.path.exists(tmp_path):
            with open(tmp_chi1_path, 'r') as f:
                self.chi1_selection = ''.join(f.readlines()[0].split('_'))
        else:
            print('Chi1 Selection file does not exist.')

        tmp_order_param = '{}_NH_Amide_Order_Parameters.pickle.gz'.format(file_name_root)
        if os.path.exists(tmp_order_param):
            self.order_parameters = pd.read_pickle(tmp_order_param)
        else:
            print('File does not exist.')

        tmp_ss_path = '{}_SecStruct_Statistics.pickle.gz'.format(file_name_root)
        if os.path.exists(tmp_ss_path):
            self.ss_stats = pd.read_pickle(tmp_ss_path)
        else:
            print('Secondary Structure Data not found.')

        tmp_rgyr = '{}_radius_of_gyration'.format(file_name_root)
        if os.path.exists(tmp_rgyr+'.pickle'):
            self.rgyr_results = _load_pickle_obj(tmp_rgyr)
        else:
            print('Radius of Gyration data not found.')

        tmp_rms = '{}_rms_quantities'.format(file_name_root)
        if os.path.exists(tmp_rms+'.pickle'):
            self.rms_data = _load_pickle_obj(tmp_rms)
        else:
            print('RMS data not found.')

        with h5py.File(file_name_root + ".hf", "r") as f:
            print("@>: cached file found: loading ", file_name_root + '.hf')
            for key in f.keys():
                print("@>:", key, f[key].dtype, f[key].shape, f[key].size)

                if f[key].size > 1:
                    # Stores value in object
                    setattr(self, key, np.zeros(f[key].shape, dtype=f[key].dtype))

                    f[key].read_direct(getattr(self, key))
                else:
                    # For a *scalar* H5Py Dataset, we index using an empty souple.
                    setattr(self, key, f[key][()])

        if self.dihed_angles is not None:
            self.dihed_angles = {'Phi': self.dihed_angles['Phi'], 'Psi': self.dihed_angles['Psi']}

        return

    ######################## Save Class ######################################
    def save_class(self, file_name_root):

        def _save_pickle_obj(obj, name):
            with open('{}.pickle'.format(name), 'wb') as f:
                pickle.dump(obj, f)
            return

        if self.atom_type_selection is not None:
            with open(file_name_root + '_sel_str.txt', 'w+') as f:
                f.write('_'.join(self.atom_type_selection))
            f.close()

        if self.chi1_selection is not None:
            with open(file_name_root + '_chi1_sel_str.txt', 'w+') as f:
                f.write('_'.join(self.chi1_selection))
            f.close()

        if self.order_parameters is not None:
            self.order_parameters.to_pickle('{}_NH_Amide_Order_Parameters.pickle.gz'.format(file_name_root))

        if self.ss_stats is not None:
            self.ss_stats.to_pickle('{}_SecStruct_Statistics.pickle.gz'.format(file_name_root))

        if self.rgyr_results is not None:
            _save_pickle_obj(self.rgyr_results, '{}_radius_of_gyration'.format(file_name_root))

        if self.rms_data is not None:
            _save_pickle_obj(self.rms_data, '{}_rms_quantities'.format(file_name_root))

        # Opens the HDF5 file and store all data.
        with h5py.File(file_name_root + ".hf", "w") as f:
            if self.chi1_angles is not None:
                f_chi1_angles = f.create_dataset("chi1_angles",
                                                        shape=self.chi1_angles.shape,
                                                        dtype=self.chi1_angles.dtype,
                                                        data=self.chi1_angles)

            if self.dihed_angles is not None:
                f_phi_angles = f.create_dataset("phi_angles",
                                                        shape=self.dihed_angles['Phi'].shape,
                                                        dtype=self.dihed_angles['Phi'].dtype,
                                                        data=self.dihed_angles['Phi'])

                f_psi_angles = f.create_dataset("psi_angles",
                                                        shape=self.dihed_angles['Psi'].shape,
                                                        dtype=self.dihed_angles['Psi'].dtype,
                                                        data=self.dihed_angles['Psi'])
