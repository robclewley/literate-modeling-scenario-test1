"""
studycontext-metadata:
  ID: 6
  tag: visualization optimized solution
"""

"""
studycontext-header:
  tag: SC_imports
"""
import PyDSTool as dst
from PyDSTool.Toolbox import phaseplane as pp
import numpy as np
from matplotlib import pyplot as plt
from common import *


test30 = cache_it['test30']
"""
studycontext-step-diagnostic:
  tag: Check quality of results
"""
viz_errors(mesh_pts_test30, test30.errors, 1)
fig1 = plt.figure(1)
fig1.savefig('viz_errors2.png')

# LFmesh was cached, so must refresh
cache_it.expire_by_object(LFmeshes5)
lin.set(pars=cache_it['pars_final'])
LFmeshes5 = Fmesh(xmesh5, ymesh5, LF)
viz_VF(LFmeshes5, (xmesh5, ymesh5), 2, 'k')
viz_VF(Fmeshes5, (xmesh5, ymesh5), 2, 'r')

fig2 = plt.figure(2)
fig2.savefig('viz_VF_overlay2.png')

"""
studycontext-footer:
  tag: MPL display
"""
plt.show()