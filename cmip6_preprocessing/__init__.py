# from importlib.metadata import version, PackageNotFoundError # only works for python 3.8 and upwards
from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("version_testing")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
    pass
