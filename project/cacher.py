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
__all__ = ['cache_it', 'exist_in_cache']


def dummy_cache(x):
    return x

# Doesn't quite help solve the 'need to cache refresh'
# problem. Only knows about existing objects!
def exist(obj):
    """
    Check if object o is different to cached version
    """
    key = pyfscache.make_digest(obj)
    return key in cache_it.get_names()

def dummy_exist(obj):
    raise NotImplementedError()

global cache_it

try:
    import pyfscache
except ImportError:
    cache_it = dummy_cache
    exist_in_cache = dummy_exist
else:
    import os
    if not os.path.isdir('./.cache'):
        os.mkdir('./.cache')
    cache_it = pyfscache.FSCache('./.cache')
    exist_in_cache = exist
