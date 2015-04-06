"""
studycontext-metadata:
  ID: 2
  tag: initial check that UAC fails
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

"""
studycontext-test:
  tag: initial test that goal fails with a small sample
"""
test = test_goal(mesh_pts_test5)
assert not test.result
print("Test failed, as expected")