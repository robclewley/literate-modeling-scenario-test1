"""
studycontext-metadata:
  ID: 3
  tag: initial visualization of linearized model's performance
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

# namespace to attach new global, shared objects
import common

__all__ = ['viz_errors', 'viz_VF']

"""
studycontext-step-diagnostic:
  tag: visualize mesh of errors
  notes:
    - figure 1
"""
test30 = test_goal(mesh_pts_test30)
viz_errors(mesh_pts_test30, test30.errors, 1)
fig1 = plt.figure(1)
fig1.savefig('viz_errors1.png')

"""
studycontext-step-diagnostic:
  tag: visualize mesh of linear vectors overlaid with actual vectors
  notes:
    - figure 2
"""

# test case
viz_VF(Fmeshes5, (xmesh5, ymesh5), 2, 'r')
viz_VF(LFmeshes5, (xmesh5, ymesh5), 2, 'k')

#viz_VF(Fmeshes30, (xmesh30, ymesh30), 2, 'r')

fig2 = plt.figure(2)
fig2.savefig('viz_VF_overlay1.png')

"""
studycontext-footer:
  tag: MPL display
"""
plt.show()