from setuptools import setup, find_packages

VERSION = '0.1.0'
DESCRIPTION = 'Best practices made easy for analysis of coorrelated motions from molecular dynamics simulations.'
LONG_DESCRIPTION = 'MDiGest is a best-practices-made-easy Python package that handles the most common issues in ' \
                   'the network-based analysis of correlated motions from molecular dynamics simulations.'
#CONTRIBUTORS = 'Gregory W. Kyro'
# Setting up
setup(
    name="mdigest",
    version=VERSION,
    author="Federica Maschietto, Brandon Allen",
    author_email="Federica Maschietto <federica.maschietto@gmail.com>, Brandon Allen <brandon.allen@yale.edu>",
    #contributors=CONTRIBUTORS,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),


    install_requires=[
        'pyemma==2.5.12', 'MDAnalysis>=2.3.0', 'silx>=1.1.1', 'numba>=0.56.4',
        'python-louvain==0.15', 'nglview>=3.0.3', 'networkx>=2.7.1'],
    keywords=['python', 'correlation', 'molecular dynamics', 'MD trajectory analysis', 'correlated motions', 'network analysis',
             'community network'],
    url="https://github.com/fmaschietto/MDiGest",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
)
