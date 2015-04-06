"""
First run of this script: You should see "In build_fn" once then
three lines of the text.

Subsequent run of this script: You should see only the three lines
of text, as they have been cached across sessions.
"""
from cacher import cache_it

@cache_it
def build_fn(pars=None):
    print("In build_fn")
    return 'some text for now'

print(build_fn())
print(build_fn())
print(build_fn())
