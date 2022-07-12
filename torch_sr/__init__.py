"""
Base module of torch-sr, PyTorch deep learning super-resolution library.
"""

try:
    from importlib import metadata
except ImportError:  # Python < 3.8 (backport)
    import importlib_metadata as metadata

__license__ = "LGPL-2.1"
__version__ = metadata.version(__package__)
__status__ = "Beta"
