import inspect
import warnings

import xarray as xr

from cmip6_preprocessing.utils import cmip6_dataset_id


# define the attrs that are needed to get an 'exact' match
exact_attrs = [
    "source_id",
    "grid_label",
    "experiment_id",
    "table_id",
    # "version", # for testing
    # "member_id",
    "variant_label",
]


def parse_metric(ds, metric, dim_length_conflict="error"):
    """Convenience function to parse a metric dataarry into an existing dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    metric : xr.Data
        Metric dataarray to add to `ds`
    dim_length_conflict : str
        Specifies how to handle dimensions that are not the same lenght.

    Returns
    -------
    xr.Dataset
        `ds` with `metric` added as coordinate.

    """
    dataset_id = cmip6_dataset_id(ds)

    if not isinstance(metric, xr.DataArray):
        raise ValueError(
            f"{dataset_id}:`metric` input must be xarray.DataArray. Got {type(metric)}"
        )

    if metric.name is None:
        warnings.warn(
            f"{dataset_id}:`metric` has no name. This might lead to problems down the line.",
            RuntimeWarning,
        )

    # Check dimension compatibility
    # TODO: Check if xarray has some nifty way to do this
    mismatch_dims = []
    for di in metric.dims:
        if len(ds[di]) != len(metric[di]):
            mismatch_dims.append(di)
    if len(mismatch_dims) > 0:
        metric_str = [str(di) + ":" + str(len(metric[di])) for di in mismatch_dims]
        ds_str = [str(di) + ":" + str(len(ds[di])) for di in mismatch_dims]
        raise ValueError(
            f"{dataset_id}:`metric` dimensions {metric_str} do not match `ds` {ds_str}",
        )

    # strip all coordinates from metric
    metric_stripped = metric.reset_coords(drop=True)

    # add attributes
    metric_stripped.attrs[
        "parsed_with"
    ] = f"cmip6_preprocessing/postprocessing/{inspect.currentframe().f_code.co_name}"
    # TODO: Get the package and module name without hardcoding. But not as important I think.

    ds = ds.assign_coords({metric_stripped.name: metric_stripped})

    return ds


def _match_attrs(ds_a, ds_b, match_attrs):
    """returns the number of matched attrs between two datasets"""
    return sum([ds_a.attrs[i] == ds_b.attrs[i] for i in match_attrs])


def match_metrics(
    ds_dict,
    metric_dict,
    match_variables,
    match_attrs=["source_id", "grid_label"],
    print_statistics=False,
    exact_attrs=exact_attrs,
):
    """Given two dictionaries of datasets, this function matches metrics from `metric_dict` to
    every datasets in `ds_dict` based on comparing the datasets attributes.

    Parameters
    ----------
    ds_dict : dict
        Dictionary of xarray datasets, that need matching metrics.
    metric_dict : dict
        Dictionary of xarray datasets, which contain metrics as data_variables.
    match_variables : list
        Data variables of datasets in `metric_dict` to match.
    match_attrs : list, optional
        Minimum dataset attributes that need to match, by default ["source_id", "grid_label"]
    print_statistics : bool, optional
        Option to print statistics about matching, by default False
    exact_attrs : list
        List of attributes that define an `exact` match, by
        default ["source_id","grid_label","experiment_id","table_id", "variant_label"].

    Returns
    -------
    dict
        All datasets from `ds_dict`, if match was not possible the input dataset is returned.

    """
    # Very rudimentary check if the data was aggregated with intake-esm
    # (which leaves non-correct attributes behind see: https://github.com/intake/intake-esm/issues/264)
    if any(["member_id" in ds.dims for ds in ds_dict.values()]):
        raise ValueError(
            "It seems like some of the input datasets in `ds_dict` have aggregated members.\
            This is not currently supported. Please use `aggregate=False` when using intake-esm `to_dataset_dict()`."
        )

    # if match is set to exact check all these attributes
    if match_attrs == "exact":
        match_attrs = exact_attrs

    total_datasets = 0
    exact_datasets = {va: 0 for va in match_variables}
    nonexact_datasets = {va: 0 for va in match_variables}
    nomatch_datasets = {va: 0 for va in match_variables}

    ds_dict_parsed = {}
    for k, ds in ds_dict.items():
        total_datasets += 1
        # Filter for the matching attrs
        matched_metric_dict = {
            kk: ds_metric
            for kk, ds_metric in metric_dict.items()
            if all([ds_metric.attrs[a] == ds.attrs[a] for a in match_attrs])
        }

        # Filter for the variable
        for mv in match_variables:
            matched_var_dict = {
                kk: ds_metric
                for kk, ds_metric in matched_metric_dict.items()
                if mv in ds_metric.variables
            }

            if len(matched_var_dict) == 0:
                warnings.warn(f"No matching metrics found for {mv}")
                nomatch_datasets[mv] += 1
            else:
                # Pick the metric that no only matches the input matches, but
                # has the highers number of matching attrs. This ensures
                # that the exact match is picked if available.

                keys = list(matched_var_dict.keys())
                nmatch_keys = [
                    _match_attrs(ds, matched_var_dict[k], exact_attrs) for k in keys
                ]
                sorted_by_match = [x for _, x in sorted(zip(nmatch_keys, keys))]

                metric_name = sorted_by_match[-1]
                ds_metric = matched_var_dict.get(metric_name)

                # this is a hardcoded check for time variable metrics.
                # These are very likely only valid for exact matches of runs.
                # For instance these could be cell thickness values, which cannot simply be used
                # for all runs of a model.
                exact_match = _match_attrs(ds, ds_metric, exact_attrs) == len(
                    exact_attrs
                )
                if "time" in ds_metric.dims and not exact_match:  # and not exact_match
                    warnings.warn(
                        "This metric had a time dimension and did not perfectly match. Not parsing anything."
                    )
                else:

                    ds_metric[mv].attrs["original_key"] = metric_name
                    ds = parse_metric(ds, ds_metric[mv])
                    if exact_match:
                        exact_datasets[mv] += 1
                    else:
                        nonexact_datasets[mv] += 1

        ds_dict_parsed[k] = ds
    if print_statistics:
        print(
            f"Processed {total_datasets} datasets.\nExact matches:{exact_datasets}\nOther matches:{nonexact_datasets}\nNo match found:{nomatch_datasets}"
        )
    return ds_dict_parsed
