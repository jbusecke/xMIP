from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("version_testing")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
    pass
