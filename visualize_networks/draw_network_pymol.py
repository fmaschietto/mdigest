import numpy as np
import warnings
from pymol import cmd, stored, selector
from pymol.cgo import *
import pandas as pd
import pickle
import networkx as nx
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def load_df(path_to_df):
    """
    Load DataFrame from pickle
    """
    try:
        return pd.read_pickle(path_to_df)
    except ValueError:
        return pickle.load(open(path_to_df, 'rb'))



def draw_network_from_df(path_to_df, reset_view=True, hide_nodes=True, **kwargs):
    """
    Draw amino-acid network from Pandas DataFrame
    """
    view = cmd.get_view()
    df = load_df(path_to_df)
    draw_network(df, **kwargs)
    if reset_view:
        cmd.set_view(view)
    if hide_nodes:
        cmd.disable('*nodes')

def isfloat(value):
    """
    Function to check that a value is a float like (convertible to float)

    Parameters
    ----------
    :param value: anything,
        value to check

    :return bool,
        True if value is convertible to float else False
    """

    if isinstance(value, list):
        value = value[0]
    try:
        float(value)
        return True
    except ValueError:
        return False


def getnum(string):
    """
    Function to get the number in a string

    Parameters
    ----------
    :param string: str,
        Input string to extract a number from

    Returns
    ----------
    :return num: int or np.nan,
        number extracted or np.nan if no number is found in the string
    """

    try:
        return int(string)
    except ValueError:
        new_str = ''
        for value in string:
            if isfloat(value):
                new_str += value
        try:
            return int(new_str)
        except ValueError:
            return np.nan


def get_best_palette(n_colors, sns_palette=None):
    if sns_palette:
        if n_colors > len(sns_palette):
            warnings.warn('Not enough colors in custom palette. We recommend choosing a palette with a '
                          'larger number of colors')
            return sns.color_palette(sns_palette, n_colors=n_colors)
        else:
            return sns_palette[:n_colors]
    if n_colors < 8:
        palette = sns.color_palette('bright', n_colors=n_colors)

    elif 8 <= n_colors <= 9:
        palette = sns.color_palette("Paired")
    else:
        palette = sns.color_palette('husl', n_colors=n_colors)

    return palette


def _color_by(df, color_by, sns_palette):
    """Bin values in column=color_by and assign different colors"""

    attributes, binEdges = np.histogram(df[color_by].values, bins=5)
    assignment = np.digitize(df[color_by], binEdges)

    n_colors = len(np.unique(assignment))+1

    palette_ = get_best_palette(n_colors, sns_palette)
    palette = []
    for a in range(0,len(assignment)):
        palette += [palette_[assignment[a]]]

    attr2color = dict(zip(df[color_by], palette))

    df.loc[:, 'color'] = df[color_by].map(attr2color)
    df.loc[:, 'color2'] = df[color_by].map(attr2color)

    return df


def draw_network(df, key=None, selection='polymer', which='weight', w1=None, w2=None, reset_view=True, default_color=(0.75, 0.75, 0.75),
                 check_nodeIDs=True, center='n. CA', label='', color_sign=False, color_by=None, sns_palette=None,
                 edge_norm=None, flattenweight=False, group_by=None, r=1):
    """
    Draw network

    Parameters
    ----------
    :param df: pd.DataFrame,
    :param key: str,
    :param selection: str,
    :param which: str
    :param w1: str or None,
    :param w2: str or None,
    :param reset_view bool,
    :param default_color: rgb set,
    :param check_nodeIDs: bool,
        if True, select nodes from df matching those in the PDB used for visualization
    :param center: str,
        where to place center of nodes
    :param label: str,
        network label
    :param color_sign: str,
        color edges by sign
    :param color_by: str,
        provide name of a colum in the df. color edges will be adjusted according to values in that column
    :param sns_palette: str,
        name of sns.color_palette
    :param edge_norm: float,
        edges weights are divided by edge_norm
    :param flattenweight: bool,
        assign weight of 0.5 to all edges having weight value other than zero
    :param group_by: None or str (name of column in df)
        group values
    :param r: float,
        radius of cilinder

    Returns
    -------

    See Also
    --------
    ``[**]`` - function adapted from https://github.com/agheeraert/pmdlearn/
    """
    if key is None:
        try:
            if 'Source' in df.columns and 'Target' in df.columns:
                key = ['Source', 'Target']
        except:
            raise NameError('Provide column name for node assignments')

    if reset_view:
        view = cmd.get_view()


    if which is None or which not in df.columns:
        raise NameError('Invalid column name.\
                         Pick one name among the column names in the data frame:\
                         {}'.format(', '.join(df.columns[2:])))
    weight = which
    if edge_norm is None:
        edge_norm = 1
    else:
        edge_norm=edge_norm

    def _check_nodeIDs(nodes_list, df_nodes):

        if len(nodes_list) == len(df_nodes):
            print('Nodes are named as increasing integers, integer list as node index')
            if isinstance(df_nodes[0], (int, np.integer)):
                return np.sort(df_nodes)
            else:
                return df_nodes
        else:
            def _popint(_):
                try:
                    return int(_)

                except (Exception,):
                    return str(getnum(_)) + ':' + _.split(':')[-1]

            nodes_intonly = pd.Series(nodes).map(_popint)
            int2nodes = dict(zip(nodes_intonly, nodes))
            nodes_idx = pd.Series(df_nodes).map(_popint).map(int2nodes)
            if all(np.array(nodes_idx) != None):
                print('Isolate resid from residue string')
                return nodes_intonly.index
            else:
                print('Failed to isolate node numbers, using index')
                return nodes_df.index

    def _draw_df(df, label=label, default_color=default_color, flattenweight=False):

        if isinstance(key, list):
            source = key[0]
            target = key[1]
            nodelist = pd.unique(df[[source, target]].values.ravel('K'))
        else:
            nodelist = pd.unique(df[key].values.ravel('K'))

        objs = []

        for index, row in df.iterrows():

            if flattenweight and row[weight] != 0:
                radius = 0.5
            else:
                radius = row[weight] / edge_norm


            if isinstance(row['color'], str):
                color = mpl.colors.to_rgb(row['color'])
            elif 'color' in row:
                color = row['color']
            else:
                color=default_color

            if 'color2' in row:
                if isinstance(row['color2'], str):
                    color2 = mpl.colors.to_rgb(row['color2'])
                else:
                    color2 = row['color2']
            else:
                color2 = color

            objs += [CYLINDER,
                     *node2CA[row[key[0]]],
                     *node2CA[row[key[1]]],
                     radius,
                     *color,
                     *color2]

        cmd.load_cgo(objs, '{}edges'.format(label))
        if isinstance(default_color, str):
            default_color = mpl.colors.to_rgb(default_color)
        obj_nodes = [COLOR, *default_color]
        for u in nodelist:
            x, y, z = node2CA[u]
            obj_nodes += [SPHERE, x, y, z, r]
        cmd.load_cgo(obj_nodes, '{}nodes'.format(label))

    #  Use pymol.stored helper variable to access defined globals are accessible from iterate-like commands such as
    #  xyz coordinates resnames, resids and chainIDs

    selection += " and {}".format(center)
    stored.posCA, stored.resnames, stored.resids, stored.chains = [], [], [], []
    cmd.iterate_state(1,selector.process(selection), "stored.posCA.append([x,y,z])")
    cmd.iterate(selection, 'stored.resnames.append(resn)')
    cmd.iterate(selection, 'stored.resids.append(resi)')
    cmd.iterate(selection, 'stored.chains.append(chain)')
    nodes = [resn+resi+':'+chain for resn, resi, chain in zip(stored.resnames, stored.resids, stored.chains)]
    nodes_df = pd.unique(df[[key[0], key[1]]].values.ravel('K'))

    if w1 is not None and w2 is not None:
        weight = '{}-{}'.format(w2, w1)
        df[weight] = df[w2] - df[w1]
    df = df.loc[(df[weight] != 0)]

    if not all(node in nodes for node in nodes_df):
        if check_nodeIDs:
            nodes = _check_nodeIDs(nodes, nodes_df)
        else:
            notin = [node for node in nodes_df if node not in nodes]
            loc = (df[key[0]].isin(notin)) | (df[key[1]].isin(notin))
            df = df.loc[~loc]

    node2CA = dict(zip(nodes, stored.posCA))


    # Color by attribute
    if color_by is not None:
        df = _color_by(df, color_by, sns_palette)

    # Color by sign of weight
    elif color_sign:
        if isinstance(color_sign, list):
            color1, color2 = color_sign
        elif color_sign == -1:
            color1, color2 = (0.20, 0.50, 0.80), (1, 0.3, 0.3)
        else:
            color1, color2 = (1, 0.3, 0.3), (0.20, 0.50, 0.80)

        print('Positive values in {} and negative values in {}'.
              format(color1, color2))

        def weight2color(X):
            return color1 if X >= 0 else color2

        df.loc[:, 'color'] = df.loc[:, weight].map(weight2color)
        df.loc[:, 'color2'] = df.loc[:, 'color']

    else:
        if 'color' not in df.columns:
            df['color'] = [default_color]  * len(df[key[0]])


    # Automatic normalization factor
    if edge_norm is None:
        edge_norm = np.max(np.abs(df[weight])) / float(r)
    else:
        edge_norm = float(edge_norm)


    # Draws groups or all or in function of sign of weight
    if group_by is not None:
        grouped = df.groupby(by=group_by)
        for grp, loc in grouped.groups.items():
            _draw_df(df.loc[loc], label=grp, flattenweight=flattenweight)
    else:
        if color_sign:
            _draw_df(df.loc[df[weight] >= 0],
                     label='pos_{}'.format(label if label != '' else weight), flattenweight=flattenweight)
            _draw_df(df.loc[df[weight] < 0], label='neg_{}'.format(label if label != '' else weight),
                     flattenweight=flattenweight)
        else:
            _draw_df(df, label=label, flattenweight=flattenweight)
            
            
def _apply_color(coefficients, selection, palette="blue_white_red"):
    stored.scores = iter(coefficients)
    if isinstance(selection, list):
        selection = ' and name CA or '.join(selection)

    cmd.alter("name CA and (not {})".format(selection), "b=0")
    cmd.alter(selection, "b=next(stored.scores)")
    cmd.spectrum("b", palette=palette, selection="name CA", byres=1)



def gradient_color(df, key="node", weight="weight", w1=None, w2=None, default_selection='name CA',
                   palette='blue_white_red', selection="polymer"):

    df = load_df(df)
    print(df)
    try:
        # if node column has format "RES1:A read resname, resid, chain"
        def to_selection(X):
            return "{} and resi {} and chain {}".\
                format(default_selection, X[3:-2], X[-1])
        nodes = df[key].map(to_selection).values

    except TypeError:
        selection += " and {}".format(default_selection)
        nodes = selection
    if w1 is None and w2 is None:
        coefficients = df[weight].values
    else:
        coefficients = df[w2].values - df[w1].values
    _apply_color(coefficients, nodes, palette=palette)



def draw_network_from_adjacency(mat, selection='polymer', center='n. CA', normalize=True, percentile=90, color='forest',
                                color2='yellowgreen', scale=1, label='', r=1):
    import itertools
    selection += " and {}".format(center)
    stored.posCA, stored.resnames, stored.resids, stored.chains = [], [], [], []
    cmd.iterate_state(1, selector.process(selection), "stored.posCA.append([x,y,z])")
    cmd.iterate(selection, 'stored.resnames.append(resn)')
    cmd.iterate(selection, 'stored.resids.append(resi)')
    cmd.iterate(selection, 'stored.chains.append(chain)')
    resindex2resid = dict(zip(np.arange(len(mat)),stored.resids))
    nodes = [resn + resi + ':' + chain for resn, resi, chain in zip(stored.resnames, stored.resids, stored.chains)]
    nodeid2CA = dict(zip(stored.resids, stored.posCA))
    node2CA = dict(zip(nodes, stored.posCA))
    pairs = list(itertools.combinations(np.arange(len(mat)), 2))
    if normalize:
        mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    if scale:
        mat *= scale
    thr = np.percentile(mat, percentile)
    print('There are %d matrix elements greater than %f' % ( len(np.where(mat>=thr)[0]), thr))

    count=0
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if j > i:
                if mat[i][j] >= thr:
                    cmd.do("distance dist%d,  resi %d and name CA, resi %d and name CA" % (count, i, j))
                    cmd.do("set dash_gap,  0, dist%d" % count)
                    cmd.do("set dash_radius, " + str(abs(mat[i, j] / 2)) + ",  dist%d" % count)
                    cmd.do('set dash_color, %s, dist%d' % (color, count))
                    count += 1
    cmd.do('hide labels')
    cmd.ray(800, 800)


def draw_shortest_path(paths_list, colors_list, weights_list=None, labels_list=None, **kwargs):
    node1, node2 = list(), list()
    for i, path in enumerate(paths_list):
        color=colors_list[i]
        node1 = path[:-1]
        node2 = path[1:]
        print("NODES", node1,node2)
        if weights_list:
            weights=weights_list[i]
        else:
            weights = np.zeros(len(node1))+1

        df = pd.DataFrame({'Source': node1,
                           'Target': node2,
                           'weight': weights})
        if labels_list:
            label='path_' + labels_list[i] + '_path' + '_%s_%s' %(node1[0], node2[-1])
        else:
            label='path_%s_%s' %(node1[0], node2[-1])

        draw_network(df, which='weight', default_color=color, label=label, **kwargs)

def draw_paths_from_df(reference_pdb, df, path_column, distance_column, avg_residues_column, output_file,
                       color_by='distance', cmap=sns.color_palette('viridis', as_cmap=True), **kwargs):


        # def normalize(x, mini=None, maxi=None):
        #     if mini is None:
        #         mini = np.min(x)
        #
        #     if maxi is None:
        #         maxi = np.max(x)
        #     return (x-mini)/(maxi-mini)
        #

        # Load the reference PDB file
        cmd.load(reference_pdb, "reference")
        
        if kwargs.__contains__('view'):
            cmd.set_view(kwargs['view'])

        # Iterate over the rows of the dataframe and visualize each path
        for index, row in df.iterrows():
            # Get the list of nodes in the path
            path_nodes = row[path_column]
            print(path_nodes)

            # Create lines connecting the CA carbons in the path
            for i in range(len(path_nodes) - 1):
                node1 = path_nodes[i]
                node2 = path_nodes[i + 1]

                # Create a unique object name for each line segment
                obj_name = f"path_{index}_line_{i}"
                # Create a new object for the line segment
                cmd.distance(obj_name, f"reference and resi {node1} and name CA",
                                   f"reference and resi {node2} and name CA")

                if color_by == 'distance':
                    # define limits for normalization. set vmin and vmax in kwargs to customize limits for normalization
                    if kwargs.__contains__('min_distance_value'):
                        min_value = kwargs['min_distance_value']
                    else:
                        min_value = np.min(df[distance_column])

                    if kwargs.__contains__('max_distance_value'):
                        max_value = kwargs['max_distance_value']
                    else:
                        max_value = np.max(df[distance_column])

                    # Create a colormap spanning the minimum-to-maximum values
                    cmap = plt.get_cmap(cmap)  # Customize the colormap if desired
                    norm = Normalize(vmin=min_value, vmax=max_value)

                    # Set the color for the line segment based on avg_residues
                    min_dist = row[distance_column]
                    normalized_value = norm(min_dist)
                    rgb_color = cmap(normalized_value)[:3]
                    cmd.set("dash_color", rgb_color, obj_name)

                    # Set the thickness of the line segments based on avg_distance value
                    avg_distance = row[distance_column]

                    cmd.set("dash_width", "%.2f" % avg_distance, obj_name)
                    cmd.set("dash_", "%.2f" % avg_distance, obj_name)
                    cmd.set("dash_gap", "0", obj_name)

                # Get the minimum and maximum values of avg_residues column
                elif color_by == 'residues_in_path':
                    if kwargs.__contains__('avg_residues_in_path'):
                        min_value = kwargs['avg_residues_in_path']
                    else:
                        min_value = np.min(df[avg_residues_column])

                    if kwargs.__contains__('avg_residues_in_path'):
                        max_value = kwargs['avg_residues_in_path']
                    else:
                        max_value = np.max(df[avg_residues_column])


                    # Create a colormap spanning the minimum-to-maximum values
                    cmap = plt.get_cmap(cmap)  # Customize the colormap if desired
                    norm = Normalize(vmin=min_value, vmax=max_value)

                    # Set the color for the line segment based on avg_residues
                    avg_residues = row[avg_residues_column]
                    normalized_value = norm(avg_residues)
                    rgb_color = cmap(normalized_value)[:3]
                    cmd.set("dash_color", rgb_color, obj_name)

                    # Set the thickness of the line segments based on avg_residues value
                    avg_residues = row[distance_column]

                    cmd.set("dash_width", "%.2f" % avg_residues, obj_name)
                    cmd.set("dash_", "%.2f" % avg_residues, obj_name)
                    cmd.set("dash_gap", "0", obj_name)


        # Center and zoom the view to see the paths clearly
        cmd.center()
        cmd.zoom()

        # Display the paths
        # pymol.cmd.show("lines", "all")

        # Save the image
        cmd.png(output_file, width=800, height=600, dpi=300)

        # Delete the temporary objects
        cmd.delete("all")


cmd.extend("draw_paths_from_df", draw_paths_from_df)
cmd.extend('gradient_color', gradient_color)
cmd.extend("draw_network_from_df", draw_network_from_df)
cmd.extend("draw_shortest_path", draw_shortest_path)
cmd.extend("draw_network_from_adjacency", draw_network_from_adjacency)
