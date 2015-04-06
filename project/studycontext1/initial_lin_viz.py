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

"""
studycontext-step:
  tag: visualize mesh of errors
  notes:
    - figure 1
"""
def viz_errors(mesh_pts, errors, fignum, scaling=2):
    fig = plt.figure(fignum)
    fig.clf()
    for pt, err in zip(mesh_pts, errors):
        plt.plot(pt[0], pt[1], 'ko', markersize=scaling*err)

test30 = test_goal(mesh_pts_test30)
viz_errors(mesh_pts_test30, test30.errors, 1)
fig1 = plt.figure(1)
fig1.savefig('viz_errors1.png')

"""
studycontext-step:
  tag: visualize mesh of linear vectors overlaid with actual vectors
  notes:
    - figure 2
"""

def viz_VF(Fmeshes, meshes, fignum, col):
    plt.figure(fignum)
    Fxmesh, Fymesh = Fmeshes
    xmesh, ymesh = meshes
    plt.quiver(xmesh, ymesh, Fxmesh, Fymesh, angles='xy', pivot='middle',
               scale=10, lw=0.5, width=0.005*max(xdom_width,ydom_width),
               minshaft=2, minlength=0.1,
               units='inches', color=col)

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