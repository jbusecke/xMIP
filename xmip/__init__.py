try:
    from importlib.metadata import (  # only works for python 3.8 and upwards
        PackageNotFoundError,
        version,
    )
except:
    from importlib_metadata import (  # works for python <3.8
        PackageNotFoundError,
        version,
    )

try:
    __version__ = version("xmip")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
    pass
