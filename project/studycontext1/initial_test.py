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
studycontext-step-test:
  tag: initial test that goal fails with a small sample
  notes:
    - store error value in cache for future reference in studycontext
"""
test = test_goal(mesh_pts_test5)
print("Max error on 5x5 mesh = {0:3f}".format(test.max_error))
cache_it['initial_error'] = test.max_error

assert not test.result
print("Test failed, as expected")