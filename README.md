# MDiGest
MDiGest Public repository.

**Best practices made easy for analysis of correlated motions from molecular dynamics simulations.**

`MDiGest` is a comprehensive and user-friendly toolbox designed to facilitate the analysis of molecular dynamics simulations. It contains a wide range of methods ranging from standard to less-standard approaches that allow users to investigate various features extracted from MD trajectories. This includes the correlated dynamics of atomic motions, diherdrals, coupled electrostatic interactions, and more, that can be used to further explore conformational changes of proteins. The tools in the package are organized in a structured way, so that users can easily integrate different metrics into their analysis. Due to the complexity of molecular dynamics analysis, the choice of method can have a major influence on the results. To support this, MDiGest allows users to easily compare multiple approaches, which benefits the user in that it constitutes an all-in one versatile and adaptable platform. Additionally, the package provides a number of visualization tools to further explore the features extracted from the MD trajectories.

## Installation

### Using pip

``pip install git+https://github.com/fmaschietto/mdigest``

### Requirements
Pip manages all dependencies but the conda environment file ``environment.yml``  can be used to 
create a new conda environment

``conda env create --name <env> --file environment.yml`` 

once the environment is created, 

``conda activate <env>`` 

will activate it.
Pymol should be installed in the environment as:

``conda install -c schrodinger pymol-bundle``

To run in a Jupyter Notebook, you will have to add this new environment to the list of kernels: 

``python -m ipykernel install --user --name=<env>``

## Getting started

### Documentation

Full documentation for the software is available in [readthedocs](https://mdigest.readthedocs.io/en/latest/)

### Hands on minimal example

Load modules

```python
    import mdigest

    from mdigest.core.parsetrajectory import *
    from mdigest.core.correlation import *
    from mdigest.core.dcorrelation import *
    from mdigest.core.networkcanvas import *
    from mdigest.core.auxiliary import *
```

load a trajectory and topology

```python
    parent = '/path/to/trajectory/'
    topology   = parent + 'a_topology.psf'
    trajectory = parent + 'a_trajectory.dcd' 
```

#### parse the trajectory by calling the MDS class in mdigest

```python
    mds = MDS()
    
    # set number of replicas
    mds.set_num_replicas(1) # use 2 if you have 2 replicas.
    
    #load topology and trajectory files into MDS class
    mds.load_system(topology, trajectory)

    #align trajectory
    mds.align_traj(inMemory=True, selection='name CA')

    set selections for MDS class
    mds.set_selection('protein and name CA', 'protein')

    #stride trajectory
    mds.stride_trajectory(initial=0, final=-1, step=5)
```

#### compute correlation from CA displacements 
```python
    dyncorr = DynCorr(mds)
    dyncorr.parse_dynamics(scale=True, normalize=True, LMI='gaussian', MI='None', DCC=True, PCC=True, VERBOSE=True, COV_DISP=True)
```

#### compute correlation from dihedrals fluctuations 
```python
    dihdyncorr = DynCorr(mds)
    dihdyncorr.parse_dih_dynamics(mean_center=True, LMI='gaussian', MI='knn_5_2', DCC=True, PCC=True, COV_DISP=True)
```
#### save for later use
```python
    savedir =  '/save/directory'
    dyncorr.save_class(file_name_root=savedir + 'dyncorr')
    dihdyncorr.save_class(file_name_root=savedir + 'dihdyncorr')
```
#### load
```python
    dyncorr_load = sd.MDSdata()
    dyncorr_load.load_from_file(file_name_root=savedir + 'dyncorr')
    dyncorr_load.load_from_file(file_name_root=savedir + 'dihdyncorr')
```
#### prepare correlation network for visualization
```python
    dist   = dyncorr_load.distances_allreplicas['rep_0'].copy() 
```
#### load different correlation matrices linearized mutual-information based generalized correlation coefficient ()
```python 
    viznetdir = '/directory/where/to/save/networks'  
    gcc    = dyncorr_load.gcc_allreplicas['rep_0']['gcc_lmi'].copy()
    dgcc   = dyncorr_load.dih_gcc_allreplicas['rep_0']['gcc_lmi'].copy()
    matrix_dictionary = {'gcc': gcc, 'dgcc':dgcc}

    vizcorr = ProcCorr()
    vizcorr.source_universe(mds.mda_u)
    vizcorr.writePDBforframe(0, viznetdir + 'frame0')
    vizcorr.set_outputparams({'outdir': viznetdir })
    vizcorr.load_matrix_dictionary(matrix_dictionary.copy())
    vizcorr.populate_attributes(matrix_dictionary.copy())
    vizcorr.set_thresholds(prune_upon=np.asarray(dist.copy()), lower_thr=0, upper_thr=5.)
    vizcorr.filter_by_distance(matrixtype='gccT', distmat=True)
    vizcorr.filter_by_distance(matrixtype='dgcc', distmat=True)
    df = vizcorr.df

    to_pickle(df, output= viznetdir + 'network_filter_d_0_5.pkl'.format(0,5))
```
#### Open Pymol in the `visualize_networks` folder 

``` python 
    cd ./mdigest/visualize_networks/
```
execute pymol locally calling `pymol` from inside the directory.
load a pdb of one frame of the system. It is best to use one frame extracted from 
the trajectory to ensure consistency with residue numbers.

```python
    from pymol import cmd, util
    import seaborn as sns

    cmd.delete('all')
    viznetdir = '/directory/where/to/save/networks'
    cmd.load(path + 'prot.pdb', '1u2p')
    cmd.color('grey80', 'prot')
    cmd.remove('!(polymer)') 
    cmd.run('draw_network_pymol.py')
    cmd.hide('lines', '*')
```
visualize short-range correlations from CA displacements on the protein

```python
    draw_network_from_df(viznetdir +'network_filter_d_0_5.pkl', which='gcc', color_by='gcc', sns_palette=sns.color_palette("tab20"), label='gcc', edge_norm=1)``
```
interactively compare with short-range correlations computed from dihedrals 
```python
    draw_network_from_df(viznetdir +'network_filter_d_0_5.pkl', which='dgcc', color_by='dgcc', sns_palette=sns.color_palette("tab20"), label='dgcc', edge_norm=1)
```
easily inspect different different metrics, such as dynamical cross correlation, mutual-information based correlation...
at the desired threshold!

Many more examples are illustrated in the mdigest-tutorial-notebook (in the ``notebooks/`` folder) with four case studies to perform analysis of MD trajectories.
Notebooks are best run in google colab. 
If run locally, add jupyter-kernel to the environment 
```python
    conda install -c anaconda ipykernel
    python -m ipykernel install --user --name=<env>
```

The molecular trajectories required for the notebook are available for download at the following links

* IGPS: https://drive.google.com/drive/folders/1XK8X18NJQY-dQUrQaeCGZtSyKeaze5mr?usp=sharing
* MptpA: https://drive.google.com/drive/folders/102mgn-bvH3GazRoMTlNqaEN6tilUJqZw?usp=sharing

### Citation 
Federica Maschietto, Brandon Allen, Gregory W. Kyro, Victor S. Batista, Journal of Chemical Physics, (2023), ``in press``; MDiGest: A Python Package for Describing Allostery from Molecular Dynamics Simulations.
[preprint to be updated](http://ursula.chem.yale.edu/~batista/publications/MDiGest.pdf)

### A Note to the Users

`MDiGest` is not the first (nor will be the last) package that allows such analysis, and therefore some of the contents were implemented before in other packages.
Some of the packages such as `MDAnalysis`, `NetworkX`, etc are imported directly, others are not directly imported but were used to some extent in building `MDiGest`.

Among these a notable  recently released package antecedent is `dynetan`, graph-oriented python package to compute and anlalyze mutual-information based generalized correlation correlation from MD trajectories.
Some of the modules of `MDiGest`, namely `processtrajectory.py` and `savedata.py` are riminescent of the structure of modules performing similar tasks in `dynetan`.
Moreover, as specifically mentioned in the documentation, some accessory functions were adapted from  it, the list of which is stated below:

* ``core.toolkit.log_progress``, generates a log bar showing the progress of the computation

* ``core.toolkit.get_path``, retrieves the minimum path from a source and target node in the calculation of the shortest_path,

* ``core.toolkit.get_NGLselection_from_node``, creates an atom selection for NGLView and an atom-selection object,

* ``core.toolkit.get_selection_from_node`` , retrieves a selection string from a node (resname, resid, segid and name), returning an atom-selection object.

Another notable package is `correlationplus`, which also focuses on analysis of correlated motions from molecular dynamics simulations.
As mentioned in the documentation, the ``compute_DCC_matrix`` and  ``compute_DCC`` functions used to compute dynamical cross-correlation coefficients in MDiGest were adapted from a related function in `correlationplus`.

[dynetan](https://github.com/melomcr/dynetan) and [correlationplus](https://github.com/tekpinar/correlationplus) are released under the GPL-v3 and LGPL licenses, hence, MDiGest was released under the GPL-v3 license. In the future, we plan to change such functions, such that we will be able to release the MDiGest under a more permissive license.

`Please remember to cite the latter when using these functionalities in MDiGest!`

Another package which deserves a mention here is [`pmdlearn`](https://github.com/agheeraert/pmdlearn).
Although the main capabilities of the latter are very different from what implemented in `MDiGest`, it provides a comprehensive module for network analysis, some parts of which we adapted in `MDiGest`.




