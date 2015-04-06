"""
header
"""
from __future__ import absolute_import
import PyDSTool as dst
from PyDSTool.Toolbox import phaseplane as pp
import numpy as np

from gist.cacher import cache_it

"""
givens
"""
import gist
from gist.model import F, target_dom, compute, L2_tol

"""
reusable convenience definitions
"""
# Convenience variables
xdom = target_dom['x']
ydom = target_dom['y']
xdom_half = sum(xdom)/2
ydom_half = sum(ydom)/2
xdom_width = xdom[1]-xdom[0]
ydom_width = ydom[1]-ydom[0]
xInterval = dst.Interval('xdom', float, xdom, abseps=1e-3)
yInterval = dst.Interval('ydom', float, ydom, abseps=1e-3)

# Convenience functions
def display_traj(traj, fignum, style=None):
    """
    Show portion of a trajectory in the problem domain
    """
    pts = traj.sample()
    fig = plt.figure(fignum)
    if style is None:
        fig.ax.plt(pts['x'], pts['y'])
    else:
        fig.ax.plt(pts['x'], pts['y'], style)


# Convenience function for orientation of new users
# (not explicit part of study context)
def demo_sim():
    print target_dom
    print F((xdom_half,ydom_half))
    pts = compute((xdom_half,ydom_half), 'test')
    print len(pts)
    plt.plot(pts['x'], pts['y'])
    plt.show()

@cache_it
def build_lin():
    # make local linear system spec
    print("I'm not building this twice!")
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
    return dst.embed(dst.Generator.Vode_ODEsystem(DSargs))

lin = build_lin()

def LF(pt):
    """mirror signature of F for linear model"""
    global lin
    x, y = pt
    return lin.Rhs(0, {'x':x, 'y':y}, asarray=True)

def metric(pt):
    """L2 metric between vector fields"""
    global LF, F
    # could usefully vectorize this
    return pp.dist(LF(pt), F(pt))

def condition(m, tol):
    return m < tol


# mesh-related
@cache_it
def make_mesh_pts(N=30):
    print "I'm not making these twice!"
    xsamples = xInterval.uniformSample(xdom_width/N, avoidendpoints=True)
    ysamples = yInterval.uniformSample(ydom_width/N, avoidendpoints=True)
    xmesh, ymesh = np.meshgrid(xsamples,ysamples)
    return xmesh, ymesh, np.dstack((xmesh,ymesh)).reshape(N*N,2)

xmesh5, ymesh5, mesh_pts_test5 = make_mesh_pts(5)
xmesh30, ymesh30, mesh_pts_test30 = make_mesh_pts(30)

@cache_it
def get_all_Fs(F, mesh_pts):
    return np.array([F(pt) for pt in mesh_pts])

# store globally for later convenience
all_Fs = get_all_Fs(F, mesh_pts_test30)
all_LFs = get_all_Fs(LF, mesh_pts_test30)

@cache_it
def Fmesh(xmesh, ymesh, F):
    """
    return x, y mesh pair of vector field values for a v.f. function 'F'
    """
    dxs, dys = np.zeros(xmesh.shape, float), np.zeros(ymesh.shape, float)
    for xi, x in enumerate(xmesh[0]):
        for yi, y in enumerate(ymesh.T[0]):
            dx, dy = F((x,y))
            # note order of indices
            dxs[yi,xi] = dx
            dys[yi,xi] = dy
    return dxs, dys


Fmeshes5 = Fmesh(xmesh5, ymesh5, F)
LFmeshes5 = Fmesh(xmesh5, ymesh5, LF)

Fmeshes30 = Fmesh(xmesh30, ymesh30, F)
LFmeshes30 = Fmesh(xmesh30, ymesh30, LF)


# test-related
def error_pts(mesh_pts):
    return np.array([metric(pt) for pt in mesh_pts])

def test_goal(mesh_pts, goal_tol=L2_tol):
    errors_array = error_pts(mesh_pts)
    result = condition(np.max(errors_array), goal_tol)
    return dst.args(result=result,
                    errors=errors_array)
