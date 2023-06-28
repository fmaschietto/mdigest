"""

MDiGest v.0.1.0

__version__ = 0.1.0
__author__ = Federica Maschietto <federica.maschietto@gmail.com>, Brandon Allen <bcallen95@gmail.com>


    DESCRIPTION
    # imports             --> general imports
    # parsetrajectory     --> process trajectory
    # correlation         --> compute correlation based on atomic displacements
    # dcorrelation        --> compute correlation from dihedrals fluctuations
    # kscorrelation       --> KS analysis
    # dimreduction        --> dimensionality reduction
    # community           --> GN, LOUVAIN, communities in general
    # savedata            --> caches the output of various models
    # auxiliary           --> auxiliary functions used by multiple modules
    # toolkit             --> accessory functions
    # plots

"""



from . import imports
from . import parsetrajectory
from . import correlation
from . import dcorrelation
from . import kscorrelation
from . import networkcommunities
from . import savedata
from . import networkcanvas
#from . import dimreduction
from . import auxiliary
from . import toolkit
from . import plots


__all__ = ['parsetrajectory', 'analysis', 'correlation', 'dcorrelation',
           'kscorrelation', 'networkcommunities', # 'dimreduction',
           'savedata', 'networkcanvas', 'auxiliary', 'toolkit', 'plots']



