try:
    import intake
except ImportError:
    intake = None


def google_cmip_col():
    """A tiny utility function to point to the 'official' pangeo cmip6 cloud files."""
    if intake is None:
        raise ValueError(
            "This functionality requires intake-esm. Install with `conda install -c conda-forge intake-esm"
        )
    return intake.open_esm_datastore(
        "https://cmip6.storage.googleapis.com/pangeo-cmip6.json"
    )
