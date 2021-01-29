import intake

def google_cmip_col():
    """A tiny utility function to point to the 'official' pangeo cmip6 cloud files."""
    return intake.open_esm_datastore(
        "https://cmip6.storage.googleapis.com/pangeo-cmip6.json"
    )