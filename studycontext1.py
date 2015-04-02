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
    return xmesh, ymesh, np.dstack((xmesh,ymesh)).reshape(N*N,2)

xmesh5, ymesh5, mesh_pts_test5 = make_mesh_pts(5)
xmesh30, ymesh30, mesh_pts_test30 = make_mesh_pts(30)

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
    # could usefully vectorize this
    return pp.dist(LF(pt), F(pt))

def condition(m, tol):
    return m < tol

"""
studycontext-goal:
  tag: define goal
  notes: functions with richer diagnostic feedback about spatial errors
"""
def error_pts(mesh_pts):
    return np.array([metric(pt) for pt in mesh_pts])

def test_goal(mesh_pts, goal_tol=L2_tol):
    errors_array = error_pts(mesh_pts)
    result = condition(np.max(errors_array), goal_tol)
    return dst.args(result=result,
                    errors=errors_array)

"""
studycontext-test:
  tag: initial test that goal fails with a small sample
"""
test = test_goal(mesh_pts_test5)
assert not test.result


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
    plt.show()

test30 = test_goal(mesh_pts_test30)
##viz_errors(mesh_pts_test30, test30.errors, 1)
##fig1 = plt.figure(1)
##fig1.savefig('viz_errors1.png')

"""
studycontext-step:
  tag: visualize mesh of linear vectors overlaid with actual vectors
  notes:
    - figure 2
"""
def get_all_Fs(F, mesh_pts):
    return np.array([F(pt) for pt in mesh_pts])

# store globally for later convenience
all_Fs = get_all_Fs(F, mesh_pts_test30)
all_LFs = get_all_Fs(LF, mesh_pts_test30)

def Fmesh(xmesh, ymesh, F):
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
