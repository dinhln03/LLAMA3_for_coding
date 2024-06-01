import apache

if apache.version == (2, 2):
    from apache22.util_script import *
else:
    raise RuntimeError('Apache version not supported.')
