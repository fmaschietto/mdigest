from mdigest.utils.imports import *

# Functions to print to screen
def print_screen_logfile(string, opened_logfile):
    """
    Print string on screen and write it on logfile

    Parameters
    ----------
    string: str,
        string to print and write
    opened_logfile: file
        opened logfile
    """

    opened_logfile.write("{}\n".format(string))
    opened_logfile.flush()  # forcing the writing by flushing the buffer


###############################################################################
###### Trajectory handling functions ##########################################
###############################################################################

# Functions to reduce trajectories
def reduce_trajectory(mda_u, selection, initial=0, final=-1, step=1):
    """
    Reduce the trajectory according to the `selection` string

    Parameters
    ----------
    mda_u: mda.Universe,
    selection: str
    initial: int,
    final: int or -1,
    step: int,

    Returns
    --------
    :returns reduced: mda.Universe object
        a reduced trajectory (mdAnalysis.Universe.trajectory)
    """

    atomgroup = mda_u.select_atoms(selection)

    ids = mda_u.select_atoms(selection).atoms.ids
    # initalize coordinates array
    if final == -1:
        final = len(mda_u.trajectory)
    coordinates = np.zeros((int(np.ceil((final - initial) / step)), len(atomgroup), 3))
    count = 0
    for ts in mda_u.trajectory[initial:final:step]:
        coordinates[count, :, :] = ts.positions[ids, :]
        count += 1
    # construct new universe with selected atoms only
    reduced = mda.Merge(atomgroup).load_new(coordinates[None, :, :], order="fac")
    return reduced


def reduce_mdt_trajectory(mdt_u, selection):
    """
    Reduce the trajectory according to the selection string

    Parameters
    ----------
    mdt_u: mdtraj.Trajectory,
    selection: str,

    Returns
    -------
    reduced, mdtraj.Trajectory object
        a reduced trajectory (mdtraj.Trajectory)
    """
    reduced = mdt_u.restrict_atoms(mdt_u.topology.select(selection))

    print('@>: reduced trajectory to %d atoms' % len(mdt_u.topology.atoms))

    return reduced

# Functions to write trajectories
def write_universe(mda_u, path, filename, chainid=False, initial=0, final=-1, step=1):
    """
    Write a trajectory and corresponding pdb to file

    Parameters
    ----------
    :params trajectory: mdAnalysis.Universe.trajectory,
    :params path: str,
    :params filename: str,
    :params chainid: bool,
        if True, write chainid to pdb
    :params ititial: int,
    :params final: int or -1
    :params step: int

    Returns
    -------
    reduced: mda.Universe object,
        a reduced trajectory (mdAnalysis.Universe.trajectory)
    """

    if final == -1:
        final = len(mda_u.trajectory)

    mda_u.trajectory[0]
    with mda.Writer(path + filename + '.dcd', mda_u.atoms.n_atoms) as W:
        for ts in mda_u.trajectory[initial:final:step]:
            W.write(mda_u.atoms)
    mda_u.trajectory[0]
    with mda.Writer(path + filename + '.pdb', mda_u.atoms.n_atoms) as W:
        if chainid:
            mda_u.add_TopologyAttr('chainIDs')
            mda_u.atoms.chainIDs = mda_u.atoms.segids
        W.write(mda_u.atoms)
    W.close()