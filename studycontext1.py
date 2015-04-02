"""
studycontext-metadata:
  ID: 1
  tag: solve scenario
"""
import PyDSTool as dst
from matplotlib import pyplot as plt

"""
studycontext-givens:
  tag: import public interface to the target model
"""
from model import F, target_dom, compute, L2_tol

# Convenience variables
xdom = target_dom['x']
ydom = target_dom['y']

# Convenience function for orientation of new users
# (not explicit part of study context)
def demo_sim():
    print target_dom
    x1 = sum(xdom)/2
    y1 = sum(ydom)/2
    print F((x1,y1))
    pts = compute((x1,y1), 'test')
    print len(pts)
    plt.plot(pts['x'], pts['y'])
    plt.show()


"""
studycontext-step:
  tag: build linear model
"""

"""
studycontext-uac:
  tag: define uac
"""

"""
studycontext-goal:
  tag: define goal
"""

"""
studycontext-test:
  tag: test goal
"""



