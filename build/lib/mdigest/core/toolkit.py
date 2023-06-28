"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: fmaschietto, bcallen95"""

from mdigest.core.imports import  *

def log_progress(sequence, every=None, size=None, name='Items', userProgress=None):
    """
    Generates log progress bar

    See Also
    --------
    ``[**]`` this function was authored by Marcelo Melo as part of https://github.com/melomcr/dynetan

    """
    from ipywidgets import IntProgress, HTML, HBox, Label
    from IPython.display import display
    from numpy import mean as npmean
    from collections import deque
    from math import floor
    from datetime import datetime
    from string import Template

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = floor(float(size) * 0.005)  # every 0.5%, minimum is 1
    else:
        assert every is not None, 'sequence is iterator, set every'

    # For elapsed time
    initTime = datetime.now()
    totTime = "?"
    labTempl = Template(" (~ $min total time (min) ; $ell minutes elapsed)")

    # If provided, we use the objects already created.
    # If not provided, we create from scratch.
    if userProgress is None or userProgress == []:

        progress = IntProgress(min=0, max=1, value=1)

        label = HTML()
        labelTime = Label("")

        box = HBox(children=[label, progress, labelTime])

        if userProgress == []:
            userProgress.append(box)
        display(box)
    else:
        box = userProgress[0]

    if is_iterator:
        # progress = IntProgress(min=0, max=1, value=1)
        box.children[1].min = 0
        box.children[1].max = 1
        box.children[1].value = 1
        box.children[1].bar_style = 'info'
    else:
        # progress = IntProgress(min=0, max=size, value=0)
        box.children[1].min = 0
        box.children[1].max = size
        box.children[1].value = 0

        # For remaining time estimation
        deltas = deque()
        lastTime = None
        meandelta = 0

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    box.children[0].value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    box.children[1].value = index
                    box.children[0].value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )

                    # Estimates remaining time with average delta per iteration
                    # Uses (at most) the last 30 iterations
                    if len(deltas) == 101:
                        deltas.popleft()

                    if lastTime:
                        deltas.append((datetime.now() - lastTime).total_seconds())
                        meandelta = npmean(deltas) / 60.0  # From seconds to minute
                        totTime = round(meandelta * size / float(every), 3)  # Mean iteration for all iterations
                    else:
                        totTime = "?"  # First iteration has no time

                    lastTime = datetime.now()

                # All ellapsed time in minutes
                elapsed = round((datetime.now() - initTime).total_seconds() / 60.0, 3)

                box.children[2].value = labTempl.safe_substitute({"min": totTime,
                                                                  "ell": elapsed})

            yield record
    except:
        box.children[1].bar_style = 'danger'
        raise
    else:
        box.children[1].bar_style = 'success'
        box.children[1].value = index
        box.children[0].value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )


def dump(filepath, array_input):
    """
    Dump np.ndarray to file

    Parameters
    ----------
    filepath: str,
        output path

    array_input: np.ndarray,
        array to save

    Returns
    -------
    pickle binary output file
    """

    file_object = open(filepath, 'wb')
    pickle.dump(array_input, file_object)
    file_object.close()


def retrieve(filepath):
    """
    Retrive pickle object

    Parameters
    ----------
    filepath: str,
        path of file to read

    Returns
    -------
    :return content: np.ndarray,
        content
    """
    if os.path.exists(filepath) == 0:
        return None
    print("@>: loading pickle file...")
    file_object = open(filepath, 'rb')
    content = pickle.load(file_object)
    file_object.close()
    return content


def file_exists(filepath):
    """
    Check if file exists

    Parameters
    ----------
    filepath: str,
        path to file

    Returns
    -------
    :return file: bool,
        whether file is in path
    """
    from os.path import exists
    file = exists(filepath)
    return file


def folder_exists(path_to_folder):
    """
    Check if directory exists, create if not

    Parameters
    ----------
    path_to_folder: str,
        path to folder

    """
    from os import makedirs
    try:
        makedirs(path_to_folder)
    except OSError:
        # directory already exists
        pass


def keywithmaxval(d):
    """
    Create a list of the dict's keys and values; return the key with the max value
    """
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]


def get_or_minus1(f):
    """
    Assign to minus one if index is absent
    """
    try:
        return f()
    except IndexError:
        return -1


def normalize(arr):
    """
    Normalize array dividing by the sum.
    """
    sum = np.sum(arr)
    arr = arr / sum
    return arr


def normalize_array(vals):
    """
    Normalize array between -1 and 1.
    """
    maxim = np.max(vals)
    minim = np.min(vals)
    normalized = np.zeros(len(vals))
    for c in range(len(vals)):
        normalized[c] = 2 * ((vals[c] - minim) / (maxim - minim)) - 1
    return normalized


def intersection(lst1, lst2):
    """
    Find intersection between two lists
    """
    lst3 = [list(filter(lambda x: x in lst1, sublist)) for sublist in lst2]
    return lst3


def list2dict(listoflists):
    """
    Convert list to dictionary
    """
    dictoflists = {}
    for i, l in enumerate(listoflists):
        dictoflists[i] = l
    return dictoflists


def dict2list(dictoflists):
    """
    Convert dictionary to list
    """
    listoflists = []
    for k, v in dictoflists.items():
        listoflists.append(v)
    return listoflists


def partition2dict(partition):
    """
    Convert partitions to dictionary having nodes as keys and assigned community (partition)
    as values
    """
    tmp_ = {}
    for i, p in enumerate(partition):
        for node in p:
            tmp_.update({node: i})
    partition = tmp_
    return partition


def get_NGLselection_from_node(node_idx, atomsel, atom=True):
    """
    Create an atom selection (whole residue or single atom) for NGLView and an atom-selection object.
    """
    node = atomsel.atoms[node_idx]
    if atom:
        return " and ".join([str(node.resid), node.resname, "." + node.name])
    else:
        return " and ".join([str(node.resid), node.resname])


def get_selection_from_node(i, atomsel, atom=False):
    """
    Get the selection string from a node index: resname, resid, segid and name
    and return an atom-selection object.
    ``[**]`` function adapted from https://github.com/melomcr/dynetan
    """
    i = int(i)
    if i < 0:
        raise
    resname = atomsel.atoms[i].resname
    resid = str(atomsel.atoms[i].resid)
    segid = atomsel.atoms[i].segid
    atomname = atomsel.atoms[i].name

    if atom:
        return "resname " + resname + " and resid " + resid + " and segid " + segid + " and name " + atomname
    else:
        return "resname " + resname + " and resid " + resid + " and segid " + segid


def get_path(src, trg, selected_atomnodes, preds, rep=0):
    """
    Return an np.ndarray with the list of nodes that connect src (source) and trg (target).
    ``[**]`` function adapted from https://github.com/melomcr/dynetan
    """

    src = int(src)
    trg = int(trg)

    if src == trg:
        return []

    if src not in preds[rep].keys():
        return []

    if trg not in preds[rep][src].keys():
        return []

    if get_selection_from_node(src, selected_atomnodes) == get_selection_from_node(trg, selected_atomnodes):
        return []

    path = [trg]

    while path[-1] != src:
        path.append(preds[rep][src][path[-1]])

    return np.asarray(path)