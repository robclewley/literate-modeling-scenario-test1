"""
studycontext-metadata:
  ID: 5
  tag: meta-step to check model linearization
"""

"""
studycontext-header:
  tag: SC_imports and preamble
"""
import PyDSTool as dst
from PyDSTool.Toolbox import phaseplane as pp
import numpy as np
import math
from matplotlib import pyplot as plt
import os

from common import *

# namespace to attach new global, shared objects
import common

this_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import yaml
with open(os.path.join(this_path,'setup.yml')) as f:
    setup = yaml.load(f)

# allow access to original system to diagnose properties of
# the parameter fit
pars = setup['pars']
eps = pars['eps']
a = pars['a']

"""
studycontext-step-calculation
  dependency: optimize
"""
Lpars= cache_it['pars_final']
taux = Lpars['taux']
x0 = Lpars['x0']
y0 = Lpars['y0']
xfy = Lpars['xfy']
yfx = Lpars['yfx']

# Point of effective linearization
# (bar_x0, bar_y0)
bar_x0 = math.sqrt((eps/taux+1)/3)
# 0.6083559537913503

yfx_guess = eps*eps/(-1+3*bar_x0*bar_x0)
print(x0/taux)

bar_y0 = x0*eps/taux - bar_x0 + bar_x0**3
# -0.2524665366427972

# Fixed (equilibrium) point of linear
# dynamical system
xfp=-(x0+yfx * y0)/(xfy * yfx-1)
yfp=-(y0+xfy * x0)/(xfy * yfx-1)

lin.set(pars=Lpars)

"""
studycontext-step-diagnostic
  tag: visualize mesh over a larger space
"""

xs_big = np.linspace(-2,2,50)
ys_big = np.linspace(-2,2,50)
xmesh_big, ymesh_big = np.meshgrid(xs_big, ys_big)
LFmeshes_big = Fmesh(xmesh_big, ymesh_big, LF)
viz_VF(LFmeshes_big, (xmesh_big, ymesh_big), 1, 'k')

fig1 = plt.figure(1)
fig1.savefig('viz_VF_overlay2.png')

"""
studycontext-footer:
  tag: MPL display
"""
plt.show()
