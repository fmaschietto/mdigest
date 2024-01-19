import numpy as np
import warnings
from pymol import cmd, stored, selector,util
from pymol.cgo import *
import pandas as pd
import pickle
import networkx as nx
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from collections import OrderedDict

cmd.load('pymol_settings.pml')


def load_df(path_to_df):
    """
    Load DataFrame from pickle
    """
    try:
        return pd.read_pickle(path_to_df)
    except ValueError:
        return pickle.load(open(path_to_df, 'rb'))


def normalize_coefficients(coefficients, n):
    """
    Normalize coefficients to be in range [-n, n]
    """
    return (coefficients - np.nanmin(coefficients)) / (np.nanmax(coefficients) - np.nanmin(coefficients)) * n


def _apply_color(coefficients, selection, palette="marine_white_deepsalmon", minimum=-1, maximum=+1):
    stored.scores = iter(coefficients)
    stored.scores = coefficients.tolist()  # Convert numpy array to list
    stored.index = 0  # Initialize an index variable

    if isinstance(selection, list):
        selection = ' or '.join(selection)
    # print(str(selection))
    cmd.select(selection)
    cmd.alter("{}".format(str(selection)), "b=0")
    cmd.alter("{}".format(str(selection)), "b=stored.scores[stored.index]; stored.index += 1")
    print('setting b factors from {} to {}'.format(minimum, maximum))

    cmd.spectrum("b", palette=palette, selection="name CA", byres=1, minimum=minimum, maximum=maximum)


def gradient_color(df, key="node", weight="avg_distance_rescale", w1=None, w2=None, default_selection='name CA',
                   palette='marine_white_deepsalmon', selection="polymer.protein", offset=0, set_range_from_values=False,
                   normalize=True, **kwargs):
    df = load_df(df)
    print(df.iloc[0:10])

    try:
        minimum = kwargs['minimum']
        maximum = kwargs['maximum']
        print(f'@>: setting minumum and maximum from kwargs: minimum={minimum}, maximum={maximum}')
    except KeyError:
        minimum = None
        maximum = None

    try:
        # if node column has format "RES1:A read resname, resid, chain"
        def to_selection(X):
            return "{} and {} and resi {} and chain {}". \
                format(selection, default_selection, str(int(X[3:-2]) - offset), X[-1])

        nodes = list(df[key].map(to_selection).values)

    except TypeError:
        selection += " and {}".format(default_selection)
        nodes = selection

    if w1 is None and w2 is None:
        coefficients = np.asarray(df[weight].values)
        # find positions where coefficients are nan and fill them with 0
        coefficients[np.isnan(coefficients)] = 0.0
    else:
        print(f'@>: w1={w1}, w2={w2}')
        coefficients = np.asarray(df[w2].values - df[w1].values)
        print(coefficients[0:10])
        # find positions where coefficients are nan and fill them with 0
        coefficients[np.isnan(coefficients)] = 0.0
        if normalize:
            print(coefficients[0:10])
            coefficients = normalize_coefficients(coefficients,10)
            print(coefficients[0:10], np.nanmin(coefficients), np.nanmax(coefficients))

    if minimum is not None and maximum is not None:
        _apply_color(coefficients, nodes, palette=palette, minimum=minimum, maximum=maximum)
    elif set_range_from_values:
        _apply_color(coefficients, nodes, palette=palette, minimum=np.nanmin(coefficients), maximum=np.nanmax(coefficients))
    else:
        _apply_color(coefficients, nodes, palette=palette, minimum=-1, maximum=+1)
cmd.extend('gradient_color', gradient_color)


def set_custom_bfactors(pdb_file, name, values, palette="rainbow", minimum=-10, maximum=10, cap=1, scale=100, save=False):
    """
    Load a PDB, set the B-factor column for CA atoms with the provided coefficients, and visualize.

    Args:
    - pdb_file (str): path to the PDB file
    - coefficients (list of floats): list of coefficients matching the number of CA atoms in the PDB
    - palette (str): color palette to use in PyMOL

    """
    # process data
    values[values > cap] = cap
    values[values < -cap] = -cap

    values = np.asarray(values, dtype=float) * scale
    print(values.min(), values.max())

    # Load the pdb file
    cmd.load(pdb_file, 'prot')

    # Extract all CA atoms from the protein object
    cmd.select("calphas", "prot and name CA")
    ca_count = cmd.count_atoms("calphas")

    if ca_count != len(values):
        raise ValueError(f"Number of CA atoms ({ca_count}) does not match length of coefficients ({len(values)})")

    # Update B-factor column for CA atoms
    for idx, coef in enumerate(values):
        # print(f'resid {idx+1}, b={coef}')
        #cmd.alter(f"calphas and index {idx + 1}", f"b={coef}")
        cmd.alter(f"calphas and resi {idx + 1}", f"b={coef}")

    # Recolor using the new B-factor values
    cmd.spectrum("b", palette, "calphas", minimum=minimum, maximum=maximum)
    cmd.set_name('prot', name)

    if save is not False:
        cmd.png(save, width=600, height=600, dpi=300, ray=1)
    # Deselect everything
    # cmd.delete("all")
    # cmd.deselect()
cmd.extend('set_custom_bfactors', set_custom_bfactors)


def equalize_in_bins(data_list, neg_bins=10, pos_bins=10):
    # Combine all arrays into one for the binning process
    combined_data = np.concatenate(data_list)

    # Separate combined data into negative, positive and zero values
    negatives = sorted([x for x in combined_data if x < 0])
    positives = sorted([x for x in combined_data if x > 0])
    zeros = [x for x in combined_data if x == 0]

    # Compute bin edges for negatives and positives based on quantiles
    neg_bin_edges = np.linspace(negatives[0], 0, neg_bins + 1, endpoint=True)
    pos_bin_edges = np.linspace(0, positives[-1], pos_bins + 1, endpoint=True)[1:]  # Excluding 0

    # Calculate midpoints for bins
    neg_midpoints = [(neg_bin_edges[i] + neg_bin_edges[i + 1]) / 2 for i in range(len(neg_bin_edges) - 1)]
    pos_midpoints = [(pos_bin_edges[i] + pos_bin_edges[i + 1]) / 2 for i in range(len(pos_bin_edges) - 1)]

    equalized_data_list = []

    # Process each array in the list
    for data in data_list:
        equalized_data_dict = OrderedDict()

        for idx, val in enumerate(data):
            if val < 0:
                bin_index = np.digitize(val, neg_bin_edges, right=True) - 1
                equalized_data_dict[idx] = neg_midpoints[bin_index]
            elif val > 0:
                bin_index = np.digitize(val, pos_bin_edges, right=True) - 1
                equalized_data_dict[idx] = pos_midpoints[bin_index]
            else:
                equalized_data_dict[idx] = 0

        # Sorting by key
        final = OrderedDict(sorted(equalized_data_dict.items()))
        equalized_data_list.append(np.asarray(list(final.values())))

    return equalized_data_list


def draw_interactions(pdb, dataframe, donor='donor_resid', acceptor='acceptor_resid', thickness='diff_Y435S_WT',
                       label='WT_C1', scale=0.01, **kwargs):

    # Initialize PyMOL
    #
    #cmd.finish_launching()

    # Create a new PyMOL session
    #cmd.reinitialize()
    #cmd.load('pymol_settings.pml')
    
    if 'color_params' in kwargs:
        color_params = kwargs['color_params']
    else:
        color_params = {'hbond': 'magenta', 'saltbridge': 'orange', 'pi_pi': 'green', 'pi_cation': 'yellow', 'hphob': 'blue', 'any':'grey'}


    cmd.hide('everything')
    cmd.load(pdb, label)
    cmd.do('set stick_radius, 0.4')
    cmd.alter('(polymer.protein)', 'chain="A"')
    cmd.alter('(resname LIG)', 'chain="L"')
    cmd.color('grey', 'all')
    cmd.set('cartoon_transparency', 0.5)
    cmd.do('hide cartoon')
    cmd.do('viewport 2026, 1358')
    cmd.set_view([0.214518383, -0.777954638, -0.590539396, -0.949557006, -0.024560178, -0.312586576, 0.228676990, 0.627817035, -0.743990481, 0.000000000, 0.000000000, -95.201255798, 34.104377747, 42.351482391, 35.855594635, 66.406684875, 123.995834351, 20.000000000])
    util.cnc('all')
    cmd.color('palegreen', 'resname LIG')
    
    cmd.do('hide sticks, hydrogens')
    util.cnc('resname LIG')

    try:
        df = pd.read_pickle(dataframe)
    except:
        df = pd.read_csv(dataframe)

    df = df.replace(':','_', regex=True)
    print(df)

    # Iterate through the DataFrame rows
    for _, row in df.iterrows():
        
        if row['type'] == 'hbond':
            donor_resid = f"{int(row[donor].split('-')[0][5:8])-320}"
            #print(donor_resid)
            acceptor_resid = f"{int(row[acceptor].split('-')[0][5:8])-320}"
            #print(acceptor_resid)
            donor_resid_str = f"chain {row[donor].split('-')[0][0:1]} and resid {donor_resid} and name {row[donor].split('-')[1]}"
            acceptor_resid_str = f"chain {row[acceptor].split('-')[0][0:1]} and resid {acceptor_resid} and name {row[acceptor].split('-')[1]}"

            object_name = f"{row['type']}_{row[donor]}_{row[acceptor]}"
            cmd.distance(f"{object_name}", f"{donor_resid_str}", f"{acceptor_resid_str}")
            
        else:
            if not np.isnan(row[thickness]):
                donor_resid = f"{int(row[donor][5:8])-320}"
                acceptor_resid = f"{int(row[acceptor][5:8])-320}"
                donor_resid_str = f'chain {row[donor][0:1]} and resid {donor_resid}'
                acceptor_resid_str = f'chain {row[acceptor][0:1]} and resid {acceptor_resid}'
                object_name = f"{row['type']}_{donor_resid}_{acceptor_resid}"
                cmd.do(f"select {donor_resid_str}")
                com_d = cmd.centerofmass('sele')
                cmd.do(f"select {acceptor_resid_str}")
                com_a = cmd.centerofmass('sele')
                cmd.pseudoatom(f"com_d_{object_name}", pos=com_d, name=f"com_d_{object_name}", state=1)
                cmd.pseudoatom(f"com_a_{object_name}", pos=com_a, name=f"com_a_{object_name}", state=1)
                cmd.do(f"show sphere, com_d_{object_name}")
                cmd.do(f"show sphere, com_a_{object_name}")

                if row['type'] == 'hphob':
                    cmd.set('sphere_color', 'blue', f"com_d_{object_name}")
                    cmd.set('sphere_color', 'blue', f"com_a_{object_name}")
                    cmd.set('sphere_transparency', 0.2, f"com_a_{object_name}")
                    cmd.set('sphere_transparency', 0.2, f"com_d_{object_name}")
                    cmd.set('sphere_scale', .8, f"com_d_{object_name}")
                    cmd.set('sphere_scale', .8, f"com_a_{object_name}")

                elif row['type'] == 'pi_pi':
                    cmd.set('sphere_color', 'green', f"com_d_{object_name}")
                    cmd.set('sphere_color', 'green', f"com_a_{object_name}")
                    cmd.set('sphere_transparency', 0.1, f"com_a_{object_name}")
                    cmd.set('sphere_transparency', 0.1, f"com_d_{object_name}")
                    cmd.set('sphere_scale', 1.0, f"com_d_{object_name}")
                    cmd.set('sphere_scale', 1.0, f"com_a_{object_name}")

                cmd.distance(f"{object_name}", f"com_a_{object_name}", f"com_d_{object_name}")

        cmd.show("sticks", f"chain {row[donor].split('-')[0][0:1]} and resid {donor_resid} and not hydrogens")
        util.cnc(f"chain {row[donor].split('-')[0][0:1]} and resid  {donor_resid} and not name hydrogens")
        cmd.show("sticks", f"chain {row[acceptor].split('-')[0][0:1]} and resid {acceptor_resid} and not hydrogens")
        util.cnc(f"chain {row[acceptor].split('-')[0][0:1]} and resid {acceptor_resid} and not hydrogens")
        
        if not np.isnan(row[thickness]):

            
            ew = abs(row[thickness])  # Absolute value for thickness
            if row['type'] == 'any' and row[thickness] < 0:
                cmd.set("dash_color", 'blue', f"{object_name}")
            elif row['type'] == 'any' and row[thickness] > 0:
                cmd.set("dash_color", 'red', f"{object_name}")
            else:
                cmd.set("dash_color", color_params[str(row['type'])], f"{object_name}")
            cmd.show("lines", f"{object_name}")
            cmd.disable(f"com_a_{object_name}")
            cmd.disable(f"com_d_{object_name}")    
            cmd.set("dash_width", float(ew) * float(scale), f"{object_name}")  # Adjust thickness

            if row[thickness] < 0:
                cmd.set("dash_gap", 0.5, f"{object_name}")
            else:
                cmd.set("dash_gap", 0.0, f"{object_name}")
        else:
            cmd.delete(f"{object_name}")

    # Zoom and display the structure
    cmd.zoom("all")
    cmd.enable('com_*')
    cmd.show('cartoon')
    cmd.set('cartoon_transparency', .8)
    cmd.set('stick_transparency', .2)
    #pymol.cmd.show("sticks")
    cmd.hide("labels")
    cmd.hide("lines", "not polymer.protein or resname LIG")
    cmd.hide("sticks", "not polymer.protein or resname LIG")
    cmd.hide("cartoon", "not polymer.protein or resname LIG")
    cmd.show("spheres", "not polymer.protein")
    cmd.color('yellow', 'resid 17+88+89+90+91+92+93+94')
    util.cnc('resid 17+88+89+90+91+92+93+94')
    cmd.color('smudge', 'resid 97+115+111+112+125+126+127+128+132')
    util.cnc('resid 97+115+111+112+125+126+127+128+132')

cmd.extend("draw_interactions", draw_interactions)