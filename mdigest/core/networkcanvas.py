"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: fmaschietto, bcallen95"""

import mdigest.core.auxiliary   as aux
import mdigest.core.toolkit     as tk
from   mdigest.core.imports     import *
from   mdigest.utils.pdbhandler import *
from   pymol                    import cmd
from   sklearn.preprocessing    import MinMaxScaler


class ProcCorr:
    """Process correlation matrix to make the desired arrays amenable for visualization."""
    def __init__(self):
        """
        Description
        -----------
        Process correlation matrix to make the desired arrays amenable for visualization.
        Possible actions to apply on desired correlation matrices include filtering, pruning upon distance matrix,
        dumping short-range entries, dumping long-range entries.

        Attributes
        ----------
        self.mda_u: mda.Universe object,
            mda.Universe
        self.matrix_dict: dict,
            dictionary containing all matrix entries for which to produce network
        self.pruning_distmat: False or str,
            - False: no pruning distance matrix provided
            - str: use name of distance matrix in matrix_dict
        self.atom_group_selection:  str,
            typically 'name CA', should match selection used to generate the correlation matrix
        self.lower_thr: float,
            A distance value in Angstrom unit. For example, it is good to remove
            high correlations for residues within less than 5.0 Angstrom distance
            to have a clear visualization. Default value np.min(self.distance).
        self.upper_thr: float,
            A distance value in Angstrom unit. The residues with this value or higher
            will not be visualized with PyMol or VMD. Default value is 9999.0 Angstrom.

        self.loc_factor: float,
            locality factor - applied as distance_matrix * 1/loc_factor
        self.inv_loc_factor,
            locality factor - applied as loc_factor * 1/distance_matrix
        self.df: pd.DataFrame,
            ouptut dataframe

        """
        self.mda_u = None
        self.matrix_dict = {}
        self.pruning_distmat = None
        self.atom_group_selstr = None
        self.atom_group_selection = None
        self.lower_thr = None
        self.upper_thr = None
        self.loc_factor = None
        self.inv_loc_factor = None
        self.df = None
        #### output ####

        self.outname = None
        self.outpath = None


    def source_universe(self, universe):
        """
        Source mda.Universe

        Parameters
        ----------
        universe: mda.Universe object

        """
        self.mda_u = universe


    def get_selection_fromMDS(self, MDS):
        """
        Retrieve atomstring selection

        Parameters
        ----------
        MDS: mdigest.MDS object


        """
        self.atom_group_selstr = MDS.atom_group_selstr


    def set_selection(self, atom_group_selstr):
        """
        Set atom group selection string

        Parameters
        ----------
        atom_group_selstr: str
            selection string
                example: 'name CA'
        """
        self.atom_group_selstr = atom_group_selstr
        print('@>: atom group selection string set to {}'.format(self.atom_group_selstr))


    def load_matrix_dictionary(self, matrix_dictionary):
        """
        Populate matrix_dictionary attribute with kwargs

        Parameters
        ----------
        matrix_dictionary: dict,
            dictionary with format of {'matrix_label': np.ndarray} containing
            correlation matrices to visualize
        """

        self.matrix_dict = OrderedDict(matrix_dictionary)


    def populate_attributes(self, matrixdict):
        """
        Create attributes of ProcCorr class corresponding to the keys of matrix_dictionary

        Parameters
        ----------
        matrixdict: dict,
            example: matrixdict = {'matrix_label': np.ndarray}
        """

        df = pd.DataFrame(columns=['Source', 'Target'])
        for matrixtype, matrix in matrixdict.items():
            print('@> setting {} attribute'.format(matrixtype))
            setattr(self, matrixtype, matrix)


    def set_thresholds(self, unit='au', prune_upon=False, **kwargs):
        """

        Parameters
        ----------
        unit: str,
            unit, default a.u., possible values 'nm', 'au'
        prune_upon: False or str,
            where str is the matrix_label of the array to use for pruning the correlations
        kwargs: dict,
            - lower_thr
            - upper_thr
            - loc_factor
            - inv_loc_factor
        """

        if unit == 'nm':
            factor = 10
        else:
            factor = 1

        if prune_upon is not False:
            if isinstance(prune_upon, str):
                if hasattr(self, prune_upon):
                    setattr(self, 'pruning_distmat', getattr(self, str(prune_upon)))
                else:
                    print('Warning: object has no attribute {}\nset attributes by calling retrieve_matrix() ')
            elif isinstance(prune_upon, np.ndarray):
                self.pruning_distmat = prune_upon

            if kwargs.__contains__('lower_thr'):
                self.lower_thr = kwargs['lower_thr']
            else:
                self.lower_thr = np.min(self.pruning_distmat * factor)
            print('@>: lower distance threshold is set to {}'.format(self.lower_thr))
            if kwargs.__contains__('upper_thr'):
                self.upper_thr = kwargs['upper_thr']
            else:
                self.upper_thr = np.max(self.pruning_distmat * factor)
            print('@>: upper distance threshold is set to {}'.format(self.upper_thr))

            if kwargs.__contains__('loc_factor'):
                self.loc_factor = kwargs['loc_factor']
            print('@>: loc_factor is set to {}'.format(self.loc_factor))
            if kwargs.__contains__('inv_loc_factor'):
                self.inv_loc_factor = kwargs['inv_loc_factor']
            print('@>: inv_loc_factor is set to {}'.format(self.inv_loc_factor))

        else:
            print('@>: No distance matrix has been loaded into class')
            print('@>: Load distance matrix using `retrieve_matrix(mat, matrixtype="distance")`')
            if kwargs.__contains__('lower_thr'):
                self.lower_thr = kwargs['lower_thr']
            else:
                self.lower_thr = np.min(self.pruning_distmat * factor)
            print('@>: lower distance threshold is set to {}'.format(self.lower_thr))
            if kwargs.__contains__('upper_thr'):
                self.upper_thr = kwargs['upper_thr']
            else:
                self.upper_thr = np.max(self.pruning_distmat * factor)
            print('@>: upper distance threshold is set to {}'.format(self.upper_thr))


    def filter_by_distance(self, matrixtype, distmat=False):
        """
        Zero out correlations lower than self.lower_str and higher than self.upper_str in a given
        correlation matrix. If residues are closer to each other than a certain distance
        (self.lower_thr), make these correlations zero. If residues are farther to
        each other than a certain distance (self.upper_thr), make these correlations
        also zero. This filter can be applied to select correlations falling within
        distance window of interest.

        Parameters
        ----------
        matrixtype: str,
            used to select a desired correlation matrix and also used as prefix for the output files.

        distmat: bool,
            default is False, which results in pruning based on correlation values.
            Upper and lower thresholds for pruning are set by call to `set_thresholds()`
        """

        matrix = getattr(self, matrixtype)
        dist_mat = None

        if distmat:
            dist_mat = self.pruning_distmat
            if self.loc_factor is None and self.inv_loc_factor is None:
                print('Pruning based on distance matrix values')

                if self.upper_thr is None or self.lower_thr is None:
                    self.upper_thr = 5.
                    self.lower_thr = 0.
                    print('@>: setting upper_thr and lower_thr for matrix filtering to default values')
                    print('@>: upper_thr = {}'.format(self.upper_thr))
                    print('@>: lower_thr = {}'.format(self.lower_thr))
                    self.set_thresholds()
                print(
                    f'@>: filtering correlations corresponding to inter-residue distances lower than {self.lower_thr} Angstrom and')
                print(f'@>: higher than {self.upper_thr} Angstrom')

        elif distmat == True and self.pruning_distmat is None:
            print('@>: call `set_thresholds(prune_upon=nd.array(distance_mat))`, where nd.array(distance_mat) is the'
                  'upon which to prune the selected correlation / adjancency matrices ')
            exit()

        else:
            print('Pruning based on correlation values')
            if self.upper_thr is None or self.lower_thr is None:
                self.upper_thr = 1.
                self.lower_thr = 0.5
                print('@>: setting upper_thr and lower_thr for matrix filtering to default values')
                print('@>: upper_thr = {}'.format(self.upper_thr))
                print('@>: lower_thr = {}'.format(self.lower_thr))

            print(f'@>: filtering correlations lower than {self.lower_thr}  and')
            print(f'@>: higher than {self.upper_thr}  inter-residue distances')

        if distmat:
            if self.loc_factor is None and self.inv_loc_factor is None:
                for i in range(0, len(matrix)):
                    for j in range(i + 1, len(matrix)):
                        if ((dist_mat[i, j] < self.lower_thr) or
                                (dist_mat[i, j] >= self.upper_thr)):
                            matrix[i, j] = 0.0
                            matrix[j][i] = 0.0
            else:
                if self.loc_factor is not None:
                    matrix = aux.filter_adjacency(matrix, distmat=dist_mat, loc_factor=self.loc_factor)

                elif self.inv_loc_factor is not None:
                    matrix = aux.filter_adjacency_inv(matrix, distmat=dist_mat, loc_factor=self.inv_loc_factor)

        else:
            for i in range(0, len(matrix)):
                for j in range(i + 1, len(matrix)):
                    if ((matrix[i, j] < self.lower_thr) or
                            (matrix[i, j] >= self.upper_thr)):
                        matrix[i, j] = 0.0
                        matrix[j][i] = 0.0

        self.matrix_dict[matrixtype] = matrix
        super().__setattr__(matrixtype, matrix)


    def to_df(self, normalize=False, **kwargs):
        """
        Save filtered matrices to pandas DataFrame

        Parameters
        ----------
        normalize: bool,
        kwargs: dict,
            - which: str,
                name of matrix (column) on which to apply normalization
            - to_range: range,
                range for normalization
        """

        df = pd.DataFrame(columns=['Source', 'Target'])

        for matrixtype, matrix in self.matrix_dict.items():
            tmp_df = pd.DataFrame(matrix)
            tmp_df = tmp_df.rename_axis('Source') \
                .reset_index() \
                .melt('Source', value_name=matrixtype, var_name='Target') \
                .query('Source != Target') \
                .reset_index(drop=True)

            if df.empty:
                df = pd.merge(left=df, right=tmp_df, how='outer')
            else:
                df = pd.merge(left=df, right=tmp_df)
            if normalize:
                if 'which' in kwargs.keys() and matrixtype in kwargs['which']:
                    print('@>: apply normalization on {} column'.format(matrixtype))

                    if 'to_range' in kwargs.keys():
                        print('@>: map {} column to {} range'.format(matrixtype, str(kwargs['to_range'])))
                        df[matrixtype] = MinMaxScaler(kwargs['to_range']).fit_transform(
                            np.array(df[matrixtype]).reshape(-1, 1))

                elif 'which' in kwargs.keys() and matrixtype not in kwargs['which']:
                    print('@>: skip normalization on {} column'.format(matrixtype))

                else:
                    print('@>: apply normalization on {} column'.format(matrixtype))
                    print('@>: map {} column to {} range'.format(matrixtype, str('(0, 1)')))
                    df[matrixtype] = MinMaxScaler().fit_transform(
                        np.array(df[matrixtype]).reshape(-1, 1))

        self.df = df

    def set_outputparams(self, params):
        """
        Set ouptut parameters
        """
        if params.__contains__('outpath'):
            self.outpath = params['outpath']
            print(self.outpath)
        else:
            self.outpath = './'

    def select_frame_coordinates(self, frame):
        """
        Select frame
        """
        ids = np.asarray(self.atom_group_selection.atoms.indices)
        return self.mda_u.trajectory[frame].positions[ids, :]

    def writePDBforframe(self, frame, outpdb):
        """
        Write PDB for a selected frame

        Parameters
        ----------
        frame: int,
            selected frame for which to write PDB

        outpdb: str,
            output filename
        """
        topology_df = mdatopology_to_dataframe(self.mda_u, frame=frame)
        df_to_pdb(topology_df, outpdb)

    def corrNetworkPymol(self, pdb_file, corr_matrix, pml_out_file, frame=0, selection=None,
                         lthr_filter=None, uthr_filter=None, edge_scaling=1, chainblocks=True,
                         **kwargs):
        """
        A useful practice is to inspect the shape of the correlation networks.
        This function provides a way to visualize the correlation patterns on the protein
        structure. The function saves a pml file that contains the user selected
        correlation values which can be executed in pymol to produce a png of the correlation
        ``[***]``

        Parameters
        ----------
        pdb_file: str,
            filename (path+filename+extension) of the PDB that will then be read in pymol. If the file is not found, the utils module is
            called to write a pdb file at the path specified by pdb_file. It is crucial that this pdb file corresponds to the trajectory frame used for reading the coordinates
            (specified by the variable ``frame``)
        frame: int,
            frame number
        corr_matrix: np.ndarray square matrix of floats,
            correlation matrix
        pml_out_file: str,
            output pml rootname
        selection: None or MDAnalysis AtomGroup object,
            - if None: selection is overwritten with self.atom_group_selstr
            - else takes in MDAnalysis AtomGroup.
        lthr_filter: float,
            lthr_filter and uthr_filter can be used to visualize correlations within an interval
            use lthr_filter to filter out correlation values below this value. Only correlation values greater than lthr_filter will be written to PML file for visualization
        uthr_filter: float,
            lthr_filter and uthr_filter can be used to visualize correlations within an interval
            use uthr_filter to filter out correlation values above this value. Only correlation values equal or lower than uthr_filter will be written to PML file for visualization
        edge_scaling: float,
            adjust radius of cylinders to be displayed in pymol.
        edge_scaling: float,
            multiplicative factor which can be used to scale the correlation values. Recommended values are between 0.01-2.00.
        chainblocks: bool,
            If True and universe contains multiple chains, separate file for inter and intra-chain correlations are printed out.
        **kwargs: dict
            provide a 'cmap_name'='name of colormap' to change the coloring scheme for edges. default is 'coolwarm'

        See Also
        --------
        ``[***]`` function adapted from https://github.com/tekpinar/correlationplus/blob/master/correlationplus/
        """

        if tk.file_exists(pdb_file):
            print('%s file will be used in pymol ' % pdb_file)
            pass
        else:
            print('write PDB for frame %d' % frame)
            self.writePDBforframe(frame, pdb_file)

        try:
            chainIDs = self.mda_u.select_atoms('all').atoms.chainIDs
        except ValueError:
            try:
                chainIDs = kwargs['chainIDs']
            except KeyError:
                chainIDs = np.asarray(['A'] * len(self.mda_u.select_atoms('all').atoms))
                print(chainIDs)
            self.mda_u.add_TopologyAttr('chainIDs', chainIDs)
            print(self.mda_u.select_atoms('all').atoms.chainIDs)

        if selection is None:
            self.atom_group_selection = self.mda_u.select_atoms(self.atom_group_selstr)
        else:
            self.atom_group_selection = self.mda_u.select_atoms(selection)

        ags = self.atom_group_selection

        if lthr_filter is None:
            lthr_filter = np.min(corr_matrix)
        if uthr_filter is None:
            uthr_filter = np.max(corr_matrix)

        coordinates = self.select_frame_coordinates(frame=frame)

        # prints
        print("@>: coordinates")
        print("@>: min value in correlation matrix: {0:.2f} Angstrom.".format(np.min(corr_matrix)))
        print("@>: max value in correlation matrix: {0:.2f} Angstrom.".format(np.max(corr_matrix)))
        print("@>: lower threshold: {0:.2f} Angstrom.".format(lthr_filter))
        print("@>: upper threshold: {0:.2f} Angstrom.".format(uthr_filter))

        print('@>: writing pymol file ... ')

        cmap_name = kwargs.get('cmap_name', 'coolwarm')
        cmap = plt.get_cmap(cmap_name)

        vmin = kwargs.get('vmin', 0)
        vmax = kwargs.get('vmax', 1)
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False),
                              cmap=cmap).set_clim(vmin=vmin, vmax=vmax)

        cylinder_string = ("CYLINDER,  {0:.3f}, {1:.3f}, {2:.3f},{3:.3f}, {4:.3f}, {5:.3f}, {6:.3f}, " +
                           "{7:.3f}, {8:.3f}, {9:.3f}, {10:.3f}, {11:.3f}, {12:.3f}, \n ")

        spheres_string = "show spheres, chain {} and resi {} and {}\n"

        PML = open(pml_out_file + '_all.pml', 'w')
        PML.write(f"cgo_transparency, .4\n")
        PML.write(f"load {pdb_file} \n")
        PML.write("color gray40, polymer.protein\n")
        PML.write("set sphere_scale, 0.75\n")

        spheresList = []
        for i in range(0, len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if ((corr_matrix[i, j] > lthr_filter) and
                        (corr_matrix[i, j] <= uthr_filter)):
                    spheresList.append(i)
                    spheresList.append(j)

        print('@> atom_group_selstr', self.atom_group_selstr)

        for i, item in enumerate(np.unique(spheresList)):
            PML.write(spheres_string.format(ags.atoms.chainIDs[item], ags.atoms.resids[item], selection))

        PML.write("python\n")
        PML.write("from pymol.cgo import *\n")
        PML.write("from pymol import cmd\n")
        PML.write("correlations = [ \n")

        # Iterate over the elements of the matrix and plot a colored square for each element

        for i in range(0, len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if (corr_matrix[i, j] > lthr_filter) and (corr_matrix[i, j] <= uthr_filter):
                    color = cmap(corr_matrix[i][j])
                    color = list(color)
                    PML.write(cylinder_string.format(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2],
                                                     coordinates[j, 0], coordinates[j, 1], coordinates[j, 2],
                                                     np.absolute(corr_matrix[i, j]) * edge_scaling,
                                                     color[0], color[1], color[2], color[0], color[1], color[2]))
        PML.write("]\n")
        PML.write("cmd.load_cgo(correlations,'correlations')\n")
        PML.write("cmd.set(\"cgo_line_width\",2.0,'correlations')\n")
        PML.write("python end")
        PML.close()

        chains = np.unique(ags.atoms.chainIDs)

        if (len(chains) > 1) & chainblocks:
            # write inter-chain correlations
            for chainI in chains:
                for chainJ in chains:
                    if chainI != chainJ:
                        PML = open(f"{pml_out_file}-interchain-chains-{chainI}-{chainJ}.pml", 'w')
                        PML.write(f"load {pdb_file} \n")
                        PML.write("cartoon type = tube\n")
                        PML.write("set sphere_scale, 0.75\n")
                        spheresList = []
                        for i in range(0, len(corr_matrix)):
                            for j in range(i + 1, len(corr_matrix)):
                                if (corr_matrix[i, j] > lthr_filter) and (corr_matrix[i, j] <= uthr_filter):
                                    if (ags.atoms.chainIDs[i] == chainI) and (ags.atoms.chainIDs[j] == chainJ):
                                        spheresList.append(i)
                                        spheresList.append(j)

                        for item in np.unique(spheresList):
                            PML.write(spheres_string.format(ags.atoms.chainIDs[item], ags.atoms.resids[item],
                                                            self.atom_group_selstr))
                        PML.write("python\n")
                        PML.write("from pymol.cgo import *\n")
                        PML.write("from pymol import cmd\n")
                        PML.write("correlations = [ \n")
                        for i in range(0, len(corr_matrix)):
                            for j in range(i + 1, len(corr_matrix)):
                                if (corr_matrix[i, j] > lthr_filter) and (corr_matrix[i, j] <= uthr_filter):
                                    if (ags.atoms.chainIDs[i] == chainI) and (ags.atoms.chainIDs[j] == chainJ):
                                        color = cmap(corr_matrix[i][j])
                                        color = list(color)
                                        PML.write(cylinder_string.format(coordinates[i, 0], coordinates[i, 1],
                                                                         coordinates[i, 2],
                                                                         coordinates[j, 0], coordinates[j, 1],
                                                                         coordinates[j, 2],
                                                                         np.absolute(corr_matrix[i, j]) * edge_scaling,
                                                                         color[0], color[1], color[2], color[0],
                                                                         color[1], color[2]))

                        PML.write("]\n")
                        PML.write("cmd.load_cgo(correlations,'correlations')\n")
                        PML.write("cmd.set(\"cgo_line_width\",2.0,'correlations')\n")
                        PML.write("python end")
                        PML.close()

            # write intra-chain correlations
            for chain in chains:
                PML = open(f"{pml_out_file}-intrachain-chain-{chain}.pml", 'w')
                PML.write(f"load {pdb_file} \n")
                PML.write("cartoon type = tube\n")
                PML.write("set sphere_scale, 0.75\n")
                spheresList = []
                for i in range(0, len(corr_matrix)):
                    for j in range(i + 1, len(corr_matrix)):
                        if (corr_matrix[i, j] > lthr_filter) and (corr_matrix[i, j] <= uthr_filter):
                            if (ags.atoms.chainIDs[i] == chain) and (ags.atoms.chainIDs[j] == chain):
                                spheresList.append(i)
                                spheresList.append(j)

                for item in np.unique(spheresList):
                    PML.write(spheres_string.format(ags.atoms.chainIDs[item], ags.atoms.resids[item],
                                                    self.atom_group_selstr))
                PML.write("python\n")
                PML.write("from pymol.cgo import *\n")
                PML.write("from pymol import cmd\n")
                PML.write("correlations = [ \n")
                for i in range(0, len(corr_matrix)):
                    for j in range(i + 1, len(corr_matrix)):
                        if (corr_matrix[i, j] > lthr_filter) and (corr_matrix[i, j] <= uthr_filter):
                            if (ags.atoms.chainIDs[i] == chain) and (ags.atoms.chainIDsa[j] == chain):
                                color = cmap(corr_matrix[i][j])
                                color = list(color)
                                PML.write(
                                    cylinder_string.format(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2],
                                                           coordinates[j, 0], coordinates[j, 1], coordinates[j, 2],
                                                           np.absolute(corr_matrix[i, j]) * edge_scaling,
                                                           color[0], color[1], color[2], color[0], color[1], color[2]))

                PML.write("]\n")
                PML.write("cmd.load_cgo(correlations,'correlations')\n")
                PML.write("cmd.set(\"cgo_line_width\",2.0,'correlations')\n")
                PML.write("python end")
                PML.close()


def display_community(path, sys, view, community_lookup, color_dict, outpath):
    """
    Pymol function to display communities on secondary structure

    Parameters
    ----------
    path: str,
        path to pdb
    sys: str,
        name of pdb to load (without extension)
    view: set,
        orientation matrix copied from pymol `get_view()`
            example:   view = (\
            -0.611808956,   -0.140187785,    0.778481722,\
             0.387945682,    0.804496109,    0.449760556,\
            -0.689334452,    0.577175796,   -0.437813699,\
             0.000000000,    0.000000000, -219.895217896,\
            45.563232422,   57.541908264,   47.740921021,\
            85.014022827,  354.776367188,   20.000000000 )
    community_lookup: mdigest.CMTY.nodes_communities_collect object,
        collected output from MD trajectory community analysis
    color_dict: dict,
        dictionary with color names as keys and rgb codes as values
    outpath: str,
        where to save png, format should be outpath = '/path/to/png/', png will be saved as outpath_communities.png
    """
    from pymol import cmd
    cmd.viewport(1892, 1754)
    cmd.delete('all')
    cmd.load(path + sys + ".pdb", 'pdb')
    cmd.set_view(view)
    cmd.show('cartoon', 'pdb')

    for c, nodes in community_lookup['comm_nodes'].items():
        com = np.asarray(nodes) + 1
        com = np.asarray(com, dtype=str)
        com = list(com)
        selection = ('+'.join(com).strip(''))
        selection = 'resi ' + selection
        cmd.select('curr_res', 'pdb and ' + selection)
        cmd.set_color('color_%d' % c, color_dict[c])
        cmd.color('color_%d' % c, 'curr_res')

    cmd.set_view(view)
    cmd.enable("all")
    print('@>: RAYTRACE')
    # cmd.ray(800, 800)
    cmd.png(outpath + '_communities.png', ray=1)


cmd.extend('display_community', display_community)


def ss_network(ss_stats, gcc, nodes_save_path, edges_save_path, num_sd=1.5):
    """
    Secondary structures network

    Parameters
    ----------
    ss_stats: pd.DataFrame,
        dataframe of secondary structure information, obtained from self.ss_stats
    gcc: np.ndarray of shape (nfeatures*nfeatures),
        pairwise generalized correlation coefficients
    nodes_save_path: str,
        path to save dictionary of nodes
    edges_save_path: str,
        path to save dictionary of edges
    num_sd: int,
        minimum standard deviations above the mean value for an edge between nodes to be considered as significant

    Returns
    -------
    dictionary of nodes: dict,
    dictionary of edges: dict
    """

    # assign secondary structure elements based on probability
    ss_stats['SSE'] = 0
    for i in range(0, len(ss_stats)):
        if ss_stats['% Coil'][i] > 50:
            ss_stats['SSE'][i] = 'coil'
        if ss_stats['% Helix'][i] > 50:
            ss_stats['SSE'][i] = 'helix'
        if ss_stats['% Strand'][i] > 50:
            ss_stats['SSE'][i] = 'strand'
        if ss_stats['% Strand'][i] < 50 and ss_stats['% Helix'][i] < 50 and ss_stats['% Coil'][i] < 50:
            ss_stats['SSE'][i] = 'diverse'
    ss_list = list(ss_stats['SSE'][1:])

    # create dictionary containing secondary structure group assignments
    community = {}
    community[int(f'{0}')] = list()
    num = 0
    for i in range(0, len(ss_list)):
        if ss_list[i] != ss_list[i - 1]:
            num += 1
            community[int(f'{num}')] = list()
            community[int(f'{num}')].append(i)
        else:
            community[int(f'{num}')].append(i)

    # compute edges
    edges = {}
    for i in range(0, len(ss_list)):
        for j in range(0, len(ss_list)):
            edges[f'{i}, {j}'] = gcc[i][j]

    # delete self-correlation keys and values
    for k, v in edges.copy().items():
        if v == 2.0:
            del edges[f'{k}']
    data_array = np.array(list(edges.items()))

    # create dataframe containing nodes and edges
    dataframe = pd.DataFrame(data_array, columns=['nodes', 'correlation'])
    nodes = dataframe['nodes'].str.split(',', expand=True)
    nodes = nodes.rename(columns={0: 'node 1', 1: 'node 2'})
    correlation = pd.DataFrame(dataframe['correlation'])
    df = pd.merge(nodes, correlation, left_index=True, right_index=True)
    df = df.iloc[::2]
    df = df.reset_index(drop=True)
    df['node 1'] = df['node 1'].astype('int')
    df['node 2'] = df['node 2'].astype('int')
    df['correlation'] = df['correlation'].astype('float')
    df = df.reset_index(drop=True)

    # delete correlations < 2 sd's above mean
    cutoff = np.mean(df['correlation']) + 2 * np.std(df['correlation'])
    df = df[abs(df['correlation']) >= cutoff]
    df = df.reset_index(drop=True)

    # compute node 1 groups
    node1 = []
    for i in range(0, len(df)):
        for k, v in iter(community.items()):
            if df['node 1'][i] in community[k]:
                node1.append(k)
    df = pd.merge(df, pd.DataFrame(node1), left_index=True, right_index=True)
    df = df.rename(columns={0: 'node 1 community'})

    # compute node 2 groups
    node2 = []
    for i in range(0, len(df)):
        for k, v in iter(community.items()):
            if df['node 2'][i] in community[k]:
                node2.append(k)
    df = pd.merge(df, pd.DataFrame(node2), left_index=True, right_index=True)
    df = df.rename(columns={0: 'node 2 community'})

    # delete intra-group correlations
    df = df[df['node 1 community'] != df['node 2 community']]
    df = df.reset_index(drop=True)

    # sum correlations between each secondary structure group pair
    communication = {}
    for i in range(0, len(df)):
        node1 = df['node 1 community'][i]
        node2 = df['node 2 community'][i]
        if f'{node1}, {node2}' in communication:
            communication[f'{node1}, {node2}'] += df['correlation'][i]
        else:
            communication[f'{node1}, {node2}'] = df['correlation'][i]

    # delete correlations < num_sd's above mean
    cutoff = np.mean(list(communication.values())) + num_sd * np.std(list(communication.values()))
    data = {k: v for k, v in communication.items() if abs(v) >= cutoff}

    # define nodes of significant edges
    edge_list = [k for k, v in data.items()]
    for i in range(len(edge_list)):
        edge_list[i] = edge_list[i].split(',')

    list2 = []
    for i in range(len(edge_list)):
        for j in edge_list[i]:
            list2.append(int(j))

    nodes = []
    for i in range(len(list2)):
        if str(list2[i]) not in nodes:
            nodes.append(int(list2[i]))

    comm = {}
    for i in range(len(nodes)):
        for k, v in community.items():
            if nodes[i] == k:
                comm[nodes[i]] = v

    # write out nodes
    with open(nodes_save_path, 'w') as comm_dict:
        comm_dict.write(str(comm))

    # write out edges
    with open(edges_save_path, 'w') as edge_dict:
        edge_dict.write(str(data))


def draw_electrostatic_network(communities_path, edges_path, save_path, fetch_pdb=None, pse_path=None,
                               edge_multiplier=5, color_ss=True):
    """
    Draw electrostatic network

    Parameters
    ----------
    communities_path: str,
        path to communities txt file
    edges_path: str,
        path to edges text file
    save_path: str,
        path to save pse file
    fetch_pdb: str,
        PDB ID to fetch, default=None
    pse_path: str,
        path to structural file if fetch_pdb=None, default=None
    edge_multiplier: int,
        multiplicative factor for edge widths in visualization, default=5
    color_ss: bool,
        whether to color structure by secondary structure, default=True

    Returns
    -------
    .pse, PyMOL pse file,

    """
    from pymol import cmd
    # initialize pymol
    cmd.delete('all')
    # load structure
    if fetch_pdb != None:
        cmd.fetch(fetch_pdb, type='pdb1')
    else:
        cmd.load(pse_path)
    # open communities dictionary in string form and convert to dictionary
    with open(communities_path) as f:
        nodes = eval(f.read())
        print('NODES: ', nodes)
    # open edges dictionary in string form and convert to dictionary
    with open(edges_path) as f:
        edges = eval(f.read())
        print('EDGES: ', edges)
    # get nodes of significant edges
    edges_split = {}
    for k, v in edges.items():
        edges_split[k.split(', ')[0]] = v
        edges_split[k.split(', ')[1]] = v
    # define list of nodes in significant edges
    node_list = []
    for ke, ve in edges_split.items():
        node_list.append(ke)
    # define a selection for each community, rename selections
    groups = []
    for kn, vn in nodes.items():
        if str(kn) in node_list:
            cmd.select('resid ' + str(vn[0]) + '-' + str(vn[-1]))
            name0 = 'group' + str(kn)
            cmd.set_name('sele', name0)
            groups.append(name0)
    # set visualization parameters
    # show groups as surfaces
    cmd.show('surface', str(groups)[1:-1])
    # set surface transparency
    cmd.set('transparency', 0.6)
    # set cartoon transparency
    cmd.set('cartoon_transparency', 0.2)

    # define script to compute center of mass
    def com(selection, state=None, mass=None, object=None, quiet=1, **kwargs):
        quiet = int(quiet)
        if (object == None):
            try:
                object = cmd.get_legal_name(selection)
                object = cmd.get_unused_name(object + "_COM", 0)
            except AttributeError:
                object = 'COM'
        cmd.delete(object)
        if (state != None):
            x, y, z = get_com(selection, mass=mass, quiet=quiet)
            if not quiet:
                print("%f %f %f" % (x, y, z))
            cmd.pseudoatom(object, pos=[x, y, z], **kwargs)
            cmd.show("spheres", object)
        else:
            for i in range(cmd.count_states()):
                x, y, z = get_com(selection, mass=mass, state=i + 1, quiet=quiet)
                if not quiet:
                    print("State %d:%f %f %f" % (i + 1, x, y, z))
                cmd.pseudoatom(object, pos=[x, y, z], state=i + 1, **kwargs)
                cmd.show("spheres", 'last ' + object)

    cmd.extend("com", com)

    def get_com(selection, state=1, mass=None, quiet=1):
        """
        Calculate the center of mass

        ``[*]`` function adapted from: https://github.com/Pymol-Scripts/Pymol-script-repo/blob/master/center_of_mass.py
        """
        quiet = int(quiet)
        totmass = 0.0
        state = int(state)
        model = cmd.get_model(selection, state)
        x, y, z = 0, 0, 0
        for a in model.atom:
            if (mass != None):
                m = a.get_mass()
                x += a.coord[0] * m
                y += a.coord[1] * m
                z += a.coord[2] * m
                totmass += m
            else:
                x += a.coord[0]
                y += a.coord[1]
                z += a.coord[2]
        if (mass != None):
            return x / totmass, y / totmass, z / totmass
        else:
            return x / len(model.atom), y / len(model.atom), z / len(model.atom)

    cmd.extend("get_com", get_com)
    # compute center of mass for each group
    for i in range(len(groups)):
        com(str(groups[i]))
    # define positive edge distances
    for k, v in edges.items():
        if v > 0:
            cmd.distance('d' + k.split(', ')[0] + k.split(', ')[1], 'group' + k.split(', ')[0] + '_COM',
                         'group' + k.split(', ')[1] + '_COM')
            cmd.set('dash_width', v * edge_multiplier, 'd' + k.split(', ')[0] + k.split(', ')[1])
            cmd.color('green', 'd' + k.split(', ')[0] + k.split(', ')[1])
        if v < 0:
            cmd.distance('d' + k.split(', ')[0] + k.split(', ')[1], 'group' + k.split(', ')[0] + '_COM',
                         'group' + k.split(', ')[1] + '_COM')
            cmd.set('dash_width', v * edge_multiplier, 'd' + k.split(', ')[0] + k.split(', ')[1])
            cmd.color('red', 'd' + k.split(', ')[0] + k.split(', ')[1])
    # remove dash gap
    cmd.set('dash_gap', 0)
    # hide labels
    cmd.hide('labels')
    # hide com spheres
    cmd.hide('spheres')
    # remove solvent molecules
    cmd.remove('sol')
    if color_ss == True:
        cmd.color('lightblue', 'ss h')
        cmd.color('lightmagenta', 'ss s')
        cmd.color('gray50', 'ss l+')
    # save pse file
    cmd.save(save_path)
