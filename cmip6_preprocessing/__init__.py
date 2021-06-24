try:
    from importlib.metadata import (
        version,
        PackageNotFoundError,
    )  # only works for python 3.8 and upwards
except:
    from importlib_metadata import (
        version,
        PackageNotFoundError,
    )  # works for python <3.8

try:
    __version__ = version("cmip6_preprocessing")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
    pass
