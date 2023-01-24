# Installation

## Using pip

``pip install git+https://github.com/fmaschietto/mdigest``

## Requirements
Pip manages all dependencies but the conda environment file ``env.yml``  can be used to 
create a new conda environment

``conda create --name <env> --file env.yml`` 

once the environment is created, 

``conda activate <env>`` 

will activate it.
Pymol should be installed in the environment as:

``conda install -c schrodinger pymol-bundle``


