"""
studycontext-metadata:
  ID: 3
  tag: attempt to optimize against metric
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
studycontext-def:
  tag: Optimization setup
"""
from scipy.optimize import minimize, fmin_cobyla

# Simplest and low-overhead way to set up an optimization.
# Param order is alphabetical:
parsdict0 = lin.query('pars')
parnames = dst.sortedDictKeys(parsdict0)
# ['taux', 'tauy', 'x0', 'xfy', 'y0', 'yfx']
pars0 = dst.sortedDictValues(parsdict0)

def residual_fn(p):
    lin.set(pars=dict(zip(parnames,p)))
    # fmin will take the norm for us
    return error_pts(mesh_pts_test5)

def constraint_taux(p):
    return p[0] >= 0

def constraint_tauy(p):
    return p[1] >= 0


"""
studycontext-step-diagnostic:
  tag: Report initial state of residual function
  dependency: initial_test.py
"""
print("Initial parameters:")
print(lin.query('pars'))
print("Initial max error = {0:3f}".format(cache_it['initial_error']))

"""
studycontext-step-calculation:
  tag: Naive optimization attempt with COBYLA
  notes:
    - use bounded domain to constrain taux and tauy parameters
"""
try:
    parsdict1 = cache_it['pars_final']
except KeyError:
    print("Recalculating optimal parameters")
    res = fmin_cobyla(residual_fn, pars0, cons=[constraint_taux, constraint_tauy],
            rhobeg=1.0, rhoend=0.0001,
            maxfun=300, disp=0)
    parsdict1 = lin.query('pars')
    cache_it['pars_final'] = parsdict1
else:
    lin.set(pars=parsdict1)

##res = minimize(residual_fn, pars0, method='COBYLA',
##                options={'xtol': 1e-3, 'disp': True,
##                         'constraints': \
##                          [{'type': 'ineq', 'fun': constraint_taux},
##                           {'type': 'ineq', 'fun': constraint_tauy}]
##                         }
##                )

"""
studycontext-step-diagnostic:
  tag: Report final state of parameters
"""
print("Final parameters:")
print(parsdict1)

"""
studycontext-step-test:
  tag: Does the solution meet the goal / UAC?
"""
test30 = test_goal(mesh_pts_test30)
print("Goal achieved? {0}".format(test30.result))
print("Max error on 30 x 30 mesh = {0:3f}".format(test30.max_error))

# store for diagnostic script
cache_it['test30'] = test30
