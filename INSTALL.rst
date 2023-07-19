## Installation

### Requirements 
Before installing mdigest through pip we recommend creating a clean environment with all required packages as specified by the ``environment.yml`` file,

``conda env create --name <env> --file environment.yml`` 

once the environment is created, 

``conda activate <env>`` 

will activate it.

### pip installation

Next, running

``pip install mdigest``

will install mdigest and all its dependencies in newly created environment.


To run in a Jupyter Notebook, you will have to add this new environment to the list of kernels: 

``python -m ipykernel install --user --name=<env>``