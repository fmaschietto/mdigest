# Installation

## Requirements
Pip manages all dependencies but the conda environment file ``environment.yml``  should be used to
create a new conda environment with the appropriate dependencies.

``conda create --name <env> --file environment.yml``

once the environment is created, 

``conda activate <env>`` 

will activate it.
Pymol should be installed in the environment as:

``conda install -c schrodinger pymol-bundle``

## Using pip

``pip install git+https://github.com/fmaschietto/mdigest``
