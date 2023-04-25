"""bamsalvage, retrieving tool of (especially long) reads from BAM files."""

#print('BAMSALVAGE: Imported {}'.format(__file__))

from bamsalvage._bamsalvage import *

try:
    __version__ = version(__name__)
except Exception:
    __version__="0.1.4"
    pass

# from _bamsalvage import *
    
