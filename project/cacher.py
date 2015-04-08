"""
Failsafe wrapper for cacheing objects in project.
If pyfscache dependency not available, then return
dummy ("do nothing") cache decorator.

Otherwise, make sure local '.cache' folder exists for
cacheing objects.

IMPORTANT:

To reset objects when their build function changes its
output during development, you have to delete this folder.

An improvement to the cacheing facility, which records the
keys of the objects as they are built, would allow both for
checking whether a cache item has expired and knowledge of
the appropriate key to delete for the object to be
rebuilt and recached.
"""
import inspect

__all__ = ['cache_it', 'can_cache']


def dummy_cache(x):
    return x

global cache_it

try:
    import pyfscache
except ImportError:
    can_cache = False
    cache_it = dummy_cache
else:
    can_cache = True
    import os
    if not os.path.isdir('./.cache'):
        os.mkdir('./.cache')
    cache_it = pyfscache.FSCache('./.cache')
    cache_it._suppress_set_cache_error = True
