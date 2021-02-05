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


def model_id_match(match_list, id_tuple):
    """Matches `id_tuple` to the list of tuples `exception_list`, which can contain
    wildcards (match any entry) and lists (match any entry that is in the list).

    Parameters
    ----------
    match_list : list
        list of tuples with id strings corresponding to e.g. `source_id`, `grid_label`...
    id_tuple : tuple
        single tuple with id strings.
    """
    # Check the size of tuples
    if any([len(t) != len(id_tuple) for t in match_list]):
        raise ValueError(
            "Each tuple in `match_list` must have the same number of elements as `match_id`"
        )

    match_list_checked = []
    for ml in match_list:
        ml_processed = []
        for i in range(len(ml)):
            content = id_tuple[i]
            if isinstance(content, str) and content != "*":
                content = [content]
            if id_tuple[i] == ml[i] or id_tuple[i] in ml[i] or ml[i] == "*":
                ml_processed.append(True)
            else:
                ml_processed.append(False)
        match_list_checked.append(all(ml_processed))
    return any(match_list_checked)
