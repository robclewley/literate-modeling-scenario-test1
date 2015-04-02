"""
Van der Pol equations to generate a nonlinear 2D vector field
and save a black box object
"""
from PyDSTool import *
from PyDSTool.Toolbox import phaseplane as pp
import sys

# public exports
__all__ = ['F', 'target_dom', 'compute', 'L2_tol']

# ----

import yaml
with open('setup.yml') as f:
    setup = yaml.load(f)

L2_tol = setup['error_tol']

#pars = {'eps': 1e-1, 'a': 0.5}
pars = setup['pars']
# target_dom = {'x': [-2.5, 2.5], 'y': [-2, 2]}
target_dom = setup['target_dom']
# Convenience variables
xdom = target_dom['x']
ydom = target_dom['y']
xInterval = Interval('xdom', float, xdom, abseps=1e-3)
yInterval = Interval('ydom', float, ydom, abseps=1e-3)


def build():
    icdict = {'x': pars['a'],
              'y': pars['a'] - pars['a']**3/3}
    xstr = '(y - (x*x*x/3 - x))/eps'
    ystr = 'a - x'

    #DSargs.fnspecs = {'Jacobian': (['t','x','y'],
    #                               """[[(1-x*x)/eps, 1/eps ],
    #                                    [    -1,        0   ]]""")}

    algparams = {'max_pts': 3000, 'init_step': 0.01, 'stiff': True}

    compatGens = findGenSubClasses('ODEsystem')

    class ODEComponent(LeafComponent):
        compatibleGens=compatGens
        targetLangs=targetLangs
    #    compatibleSubcomponents=(sys,)

    vdpC = ODEComponent('vdp_gen')
    vdpC.add([Var(xstr, 'x', domain=target_dom['x'], specType='RHSfuncSpec'),
              Var(ystr, 'y', domain=target_dom['y'], specType='RHSfuncSpec'),
              Par(pars['a'], 'a'), Par(pars['eps'], 'eps')])
    vdpC.flattenSpec()

    MC = ModelConstructor('vdp',
                   generatorspecs={'vdp_gen':
                        {'modelspec': vdpC,
                         'target': 'Vode_ODEsystem',
                         'algparams': algparams}
                       },
                   indepvar=('t',[0,10]),
                   eventtol=1e-5, activateAllBounds=True,
                   withStdEvts={'vdp_gen': True},
                   stdEvtArgs={'term': True,
                               'precise': True,
                               'eventtol': 1e-5},
                   abseps=1e-7,
                   withJac={'vdp_gen': True})

    # withJac option for automatic calc of Jacobian
    return MC.getModel()

try:
    vdp = loadObjects('model.sav')[0]
except:
    vdp = build()
    saveObjects(vdp, 'model.sav')


def F(xarray):
    x, y = xarray
    assert x in xInterval
    assert y in yInterval
    return vdp.Rhs(0, {'x':x, 'y':y}, asarray=True)

def compute(xarray, trajname='test'):
    x, y = xarray
    assert x in xInterval
    assert y in yInterval
    icdict = {'x': x,
              'y': y}
    vdp.set(ics=icdict)
    vdp.compute(trajname, tdata=[0,5])
    pts = vdp.sample(trajname)
    return pts

def sanity_check():
    print target_dom
    x1 = sum(xdom)/2
    y1 = sum(ydom)/2
    print F((x1,y1))
    pts = compute((x1,y1), 'test')
    print len(pts)
    plt.plot(pts['x'], pts['y'], 'k-o')

if __name__ == '__main__':
    vdp_gen = vdp.registry.values()[0]
    pp.plot_PP_vf(vdp_gen, 'x', 'y', subdomain=target_dom,
                  N=20, scale_exp=-1)
    sanity_check()
    plt.show()
