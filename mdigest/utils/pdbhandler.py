from mdigest.utils.imports import  *

def df_to_pdb(input_dataframe, desired_outname, protonation=False, charge_arr=False,
              split_by_chain=False, split_by_segid=False):
    """
    Advanced df to PDB with customization of chainIDs segIDs, and charges

    Parameters:
    input_dataframe: pd.DataFrame,
    desired_outname: str,
    protonation: bool,
    charge_arr: False or np.ndarray
    split_by_chain: bool,
    split_by_segid: bool
    """
    df_vals = input_dataframe.values

    fmt = {
        'ATOM': (
            "{record_name:6s}{serial:5d} {name:^3s}{altLoc:<1s}{resName:<4s}{chainID:1s}{resSeq:4d}{iCode:1s}   {posx:8.3f}{posy:8.3f}{posz:8.3f}{occupancy:6.2f}{tempFactor:6.2f}      {segID:<4s}{element:>2s}{charge:<2s}\n")}
    if (split_by_chain == False) and (split_by_segid == False):
        file = open(desired_outname + ".pdb", 'w')
        # print("no split")
    elif split_by_chain:
        chains = np.unique(input_dataframe['chainID']).astype(str)
        # print(chains)
        outfiles = {}
        for cID in chains:
            outfiles[cID] = [open(desired_outname + '_chain_%s.pdb' % cID, 'w')]
    elif split_by_segid:
        segids = np.unique(input_dataframe['segID']).astype(str)
        outfiles = {}
        for sID in segids:
            outfiles[sID] = [open(desired_outname + '_segid_%s.pdb' % sID, 'w')]

    for idx, line in enumerate(df_vals):
        record_name, serial, name, altloc, resname, chainid, reseq, icode, posx, posy, posz, occup, bfact, segid, atom_type, charge = line
        if charge_arr is not False:
            charge = str(charge_arr[idx])
        else:
            charge = charge

        vals = {'record_name': record_name, 'serial': serial}
        if len(name) >= 4:
            vals['name'] = name[:4]
        elif len(name) == 3:
            vals['name'] = ' {}'.format(name)
        elif len(name) == 2:
            vals['name'] = ' {} '.format(name)
        elif len(name) == 1:
            vals['name'] = ' {}  '.format(name)
        vals['altLoc'] = altloc
        vals['resName'] = resname
        vals['chainID'] = chainid
        vals['resSeq'] = reseq
        vals['iCode'] = icode
        vals['posx'] = posx
        vals['posy'] = posy
        vals['posz'] = posz  # don't take off atom so conversion works
        vals['occupancy'] = occup
        vals['tempFactor'] = bfact
        vals['segID'] = segid
        vals['element'] = atom_type
        vals['charge'] = '0'

        if vals['element'].strip(' ') == '':
            if (vals['resName'].strip(' ') == 'Cl' or vals['resName'].strip(' ') == 'CL' or vals['resName'].strip(
                    ' ') == 'Cl-'):
                vals['element'] = 'Cl'
            elif (vals['resName'].strip(' ') == 'Na' or vals['resName'].strip(' ') == 'NA' or vals['resName'].strip(
                    ' ') == 'Na+'):
                vals['element'] = 'Na'
            else:
                vals['element'] = vals['name'].strip(' ')[0]
                vals['charge'] = charge

        if protonation:
            if (vals['resName'].strip(' ') == 'ARG') and (vals['name'].strip(' ') == 'NH2'):
                vals['charge'] = '1+'
            elif (vals['resName'].strip(' ') == 'LYS') and (vals['name'].strip(' ') == 'NZ'):
                vals['charge'] = '1+'
            elif (vals['resName'].strip(' ') == 'HID') and (vals['name'].strip(' ') == 'ND1'):
                vals['charge'] = '1+'
            elif (vals['resName'].strip(' ') == 'HIE') and (vals['name'].strip(' ') == 'NE2'):
                vals['charge'] = '1+'
            elif (vals['resName'].strip(' ') == 'ASP') and (vals['name'].strip(' ') == 'OD2'):
                vals['charge'] = '1-'
            elif (vals['resName'].strip(' ') == 'GLU') and (vals['name'].strip(' ') == 'OE2'):
                vals['charge'] = '1-'
            else:
                vals['charge'] = '0'
                # .. _ATOM: http://www.wwpdb.org/documentation/file-format-content/format32/sect9.html#ATOM

        line = fmt['ATOM'].format(**vals)

        if (split_by_chain == False) and (split_by_segid == False):
            if len(line.strip(' ')) > 2:
                #print('@>',line)
                file.write(line)

        elif split_by_chain:
            if len(line.strip(' ')) > 2:
                chain = vals['chainID']
                if vals['chainID'] == chain:
                    outfiles[chain][0].write(line)
                chain = vals['chainID']

        elif split_by_segid:
            if len(line.strip(' ')) > 2:
                segid = vals['segID']
                if vals['segID'] == segid:
                    outfiles[segid][0].write(line)
                segid = vals['segID']

    if not ((split_by_chain == False) and (split_by_segid == False)):
        for k, v in outfiles.items():
            outfiles[k][0].close()
    else:
        file.close()

    return 'Your PDB File is written.'


def select_frame(mda_u, frame):
    u = mda_u.copy()
    return u.trajectory[frame]


def mdatopology_to_dataframe(mda_u, frame, selection='all'):
    df = pd.DataFrame()
    extract_element = lambda x: [e[0] for e in x]
    u = mda_u.copy()
    ag = u.select_atoms(selection)
    coordinates = select_frame(mda_u=u, frame=frame).positions
    try:
        record_types = ag.atoms.record_types
    except:
        record_types = ['ATOM'] * ag.atoms.n_atoms

    atom_serial  = np.arange(len(ag.atoms.ids))+1

    try:
        altloc   = ag.atoms.atoms.altLocs
    except:
        altloc   =  [' '] * ag.atoms.n_atoms
    resname      = ag.atoms.resnames
    try:
        icodes   = ag.atoms.icodes
    except:
        icodes   = [' '] * ag.atoms.n_atoms

    segids       = ag.atoms.segids
    try:
        chainIDs = ag.atoms.chainIDs
    except:
        chainIDs = [' '] * ag.atoms.n_atoms
    resids       = ag.atoms.resids
    coordx       = coordinates[:,0]
    coordy       = coordinates[:,1]
    coordz       = coordinates[:,2]

    try:
        occupancies  = ag.atoms.occupancies
    except:
        occupancies = np.zeros(ag.atoms.n_atoms, dtype=float)
    try:
        tempfactors  = ag.atoms.tempfactors
    except:
        tempfactors = np.zeros(ag.atoms.n_atoms, dtype=float)

    atomnames    = ag.atoms.names
    elements = extract_element(ag.atoms.names)

    df['record_names'] = record_types
    df['serial']  = atom_serial
    df['name'] = atomnames
    df['altLoc'] = altloc
    df['resName'] = resname
    df['chainID'] = chainIDs
    df['resSeq'] = resids
    df['iCode']   = icodes
    df['coordx']  = coordx
    df['coordy']  = coordy
    df['coordz']  = coordz
    df['occupancy']  = occupancies
    df['tempFactor'] = tempfactors
    df['segID']      = segids
    df['element'] = elements
    df['charge'] = np.zeros(len(resids), dtype=int)

    return df


def read_pdb_file(filename):
    out_lists = []
    with open(filename) as file:
        for line in file.readlines():
            if line.startswith('ATOM'):
                out_lists.append([line[:6].strip(' '), int(line[6:11].strip(' ')),  line[12:16].strip(' '), line[16], \
                                            line[17:20], line[21], int(line[22:26].strip(' ')),  line[26], \
                                            float(line[30:38]), float(line[38:46]), float(line[46:54]), \
                                             float(line[54:60]), float(line[60:66]), line[72:76].strip(' '), \
                                             line[76:78].strip(' '), [i if len(i) > 0 else '0' for i in line[78:80]][0]])
    column_names = ['Record Name', 'Atom Serial No.', 'Atom Name', 'Alt. Loc.', 'Res. Name', 'Chain ID',
                    'Res. No.', 'Ins. Code', 'posx', 'posy', 'posz', 'Occ.', 'B Factor', 'Seg. ID', 'Atom Type', 'Charge']
    out_data = pd.DataFrame(out_lists, columns = column_names)
    return out_data


def center_pdb(pdb_df, shift):
    pdb_df['posx'] = pdb_df['posx'].apply(lambda x: -x + shift[0] / 2)
    pdb_df['posy'] = pdb_df['posy'].apply(lambda x: -x + shift[0] / 2)
    pdb_df['posz'] = pdb_df['posz'].apply(lambda x: -x + shift[0] / 2)
    return pdb_df


def fill_topology_df(topo_df_mdt, coord_array):
    """
    This function completes the topology data frame from MDtraj for later
    saving the topology to PDB format.

    topo_df_mdt: topology from mdtraj.to_dataframe()[0]:
                      which only contains the following columns:
                      ['serial', 'name', 'element', 'resSeq', 'resName', 'chainID','segmentID']

    coord_array: array of coordinates from mdtraj.trajectory.xyz or average_structure.xyz

    Returns
    -------
    :return topo_df_mdt: topology dataframe completed of all fields
    """

    topo_df_mdt['record_name'] = ['ATOM']*len(topo_df_mdt)
    topo_df_mdt['altLoc'] = [' ']*len(topo_df_mdt)
    topo_df_mdt['posx'] = coord_array[:,0]
    topo_df_mdt['posy'] = coord_array[:,1]
    topo_df_mdt['posz'] = coord_array[:,2]

    # relpace chainID with alphabet letters in topo_df['chainID'] column
    topo_df_mdt['chainID'] = [chr(int(i)+65) for i in topo_df_mdt['chainID']]

    #rename colum of dataframe from segmentID to segID
    topo_df_mdt = topo_df_mdt.rename(columns={'segmentID':'segID'})

    # fill in the missing values of segID with chainID
    topo_df_mdt['segID'] = topo_df_mdt['chainID']
    topo_df_mdt['occupancy'] = [0.0]*len(topo_df_mdt)
    topo_df_mdt['tempFactor'] = [0.0]*len(topo_df_mdt)
    topo_df_mdt['iCode'] = [' ']*len(topo_df_mdt)
    topo_df_mdt['charge'] = [' ']*len(topo_df_mdt)

    # reorganize columns in topo_df dataframe according to PDB format
    col_names = ['record_name', 'serial', 'name', 'altLoc', 'resName', 'chainID', 'resSeq', 'iCode',
                 'posx', 'posy', 'posz', 'occupancy', 'tempFactor', 'segID', 'element', 'charge']

    # reorder columns in topo_df by names in column_names
    topo_df_mdt = topo_df_mdt[col_names]
    return topo_df_mdt

