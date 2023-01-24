# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

print(sys.executable)
for x in os.walk('../../'):
  sys.path.insert(0, x[0])
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mdigest'
copyright = '2022, Federica Maschietto, Brandon Allen'
author = 'Federica Maschietto; Brandon Allen'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
             'sphinx_rtd_theme',
             'sphinx.ext.napoleon']

templates_path = ['_templates']

exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']

# Autodoc mock imports
# autodoc_mock_imports = ["numpy","np","cython","MDAnalysis","mda", "pyemma", "scipy"
#                         "networkx","nx","community", "operator", "collections",
#                         "pickle", "h5py","numba", "pandas", "pd", "mdtraj", "md", "sklearn",
#                         "sklearn.neighbors", "sklearn.preprocessing", "scipy.special", "tqdm",
#                         "operator.itemgetter", "concurrent.futures.thread",
#                         "MDAnalysis.coordinates.memory",
#                         "MDAnalysis.analysis.base",
#                         "MDAnalysis.analysis",
#                         "MDAnalysis.analysis.dihedrals",
# 		        "MDAnalysis.analysis.distances"]
# 

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
 
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


