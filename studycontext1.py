"""
studycontext-metadata:
  ID: 1
  tag: solve scenario
"""
import PyDSTool as dst
from PyDSTool.Toolbox import phaseplane as pp
import numpy as np
from matplotlib import pyplot as plt

"""
studycontext-givens:
  tag: import public interface to the target model
"""
from model import F, target_dom, compute, L2_tol

# Convenience variables
xdom = target_dom['x']
ydom = target_dom['y']
xdom_half = sum(xdom)/2
ydom_half = sum(ydom)/2
xdom_width = xdom[1]-xdom[0]
ydom_width = ydom[1]-ydom[0]
xInterval = dst.Interval('xdom', float, xdom, abseps=1e-3)
yInterval = dst.Interval('ydom', float, ydom, abseps=1e-3)

# Convenience function for orientation of new users
# (not explicit part of study context)
def demo_sim():
    print target_dom
    print F((xdom_half,ydom_half))
    pts = compute((xdom_half,ydom_half), 'test')
    print len(pts)
    plt.plot(pts['x'], pts['y'])
    plt.show()


"""
studycontext-step:
  tag: build linear model
  notes:
    - define build function
"""
def build_lin():
    # make local linear system spec
    DSargs = dst.args(name='lin')
    xfn_str = '(x0+yfx*y - x)/taux'
    yfn_str = '(y0+xfy*x - y)/tauy'
    DSargs.varspecs = {'x': xfn_str, 'y': yfn_str}
    DSargs.xdomain = {'x': xdom, 'y': ydom}
    DSargs.pars = {'x0': xdom_half, 'y0': ydom_half,
                   'xfy': 1, 'yfx': 1,
                   'taux': 1, 'tauy': 1}
    DSargs.algparams = {'init_step':0.001,
                        'max_step': 0.001,
                        'max_pts': 10000}
    DSargs.checklevel = 0
    DSargs.tdata = [0, 10]
    DSargs.ics = {'x': xdom_half*1.1, 'y': ydom_half*1.1}
    DSargs.fnspecs = {'Jacobian': (['t', 'x', 'y'],
                                   """[[-1/taux, -yfx/taux],
                                       [-xfy/tauy, -1/tauy]]""")}
    return dst.Generator.Vode_ODEsystem(DSargs)

"""
    - instantiate local model
"""
lin = build_lin()

"""
studycontext-uac:
  tag: define uac
  notes:
    - test sample 30 x 30 grid of ICs in domain
"""
def make_mesh_pts(N=30):
    xsamples = xInterval.uniformSample(xdom_width/N, avoidendpoints=True)
    ysamples = yInterval.uniformSample(ydom_width/N, avoidendpoints=True)
    xmesh, ymesh = np.meshgrid(xsamples,ysamples)
    return np.dstack((xmesh,ymesh)).reshape(N*N,2)

mesh_pts_test5 = make_mesh_pts(5)
mesh_pts_test30 = make_mesh_pts(30)

"""
    - mirror signature of F for linear model
"""
def LF(pt):
    x, y = pt
    return lin.Rhs(0, {'x':x, 'y':y})

"""
    - L2 metric between vector fields
"""
def metric(pt):
    return pp.dist(LF(pt), F(pt))

def condition(m, tol):
    return m < tol

"""
studycontext-goal:
  tag: define goal
  notes: first pass, naive function with no diagnostic feedback about spatial errors
"""
def test_goal(mesh_pts, goal_tol=L2_tol):
    max_err = np.Inf
    for pt in mesh_pts:
        err =  metric(pt)
        if err < max_err:
            max_err = err
    return condition(max_err, goal_tol)

"""
studycontext-test:
  tag: initial test that goal fails with a small sample
"""
assert not test_goal(mesh_pts_test5)




