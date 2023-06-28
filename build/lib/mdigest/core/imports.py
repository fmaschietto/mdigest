"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: fmaschietto, bcallen95"""

import pandas           as pd
import numpy            as np
import networkx         as nx
import mdtraj           as md
import MDAnalysis       as mda
import pickle
import h5py
import scipy
import os
import sklearn
#import pyemma
from itertools import combinations_with_replacement as cwr
from tqdm import tqdm
from tqdm.notebook import trange
from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.cm as cm
import seaborn as sns
import warnings

# matplotlib parameters
plt.rcParams['axes.labelpad'] = 3.
plt.rcParams['font.size'] = 30
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 10
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = "Dejavu Sans"
plt.rcParams['font.serif'] = "Dejavu Sans"
plt.rcParams['figure.autolayout'] = True

