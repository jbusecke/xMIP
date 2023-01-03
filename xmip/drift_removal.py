import warnings

import dask.array as dsa
import numpy as np
import xarray as xr
import xarrayutils as xru

from xarrayutils.utils import linear_trend

from xmip.postprocessing import EXACT_ATTRS, _match_datasets
from xmip.utils import cmip6_dataset_id


def _maybe_unpack_date(date):
    """`Unpacks` cftime from xr.Dataarray if provided"""
    # I should probably not do this here but instead in the higher level functions...
    if isinstance(date, xr.DataArray):
        date = date.data.tolist()
        if isinstance(date, list):
            if len(date) != 1:
                raise RuntimeError(
                    "The passed date has the wrong format. Got [{date}] after conversion to list."
                )
            else:
                date = date[0]
    return date


def _construct_cfdate(data, units, calendar):
    # This seems clunky. I feel there must be a more elegant way of doing this?
    date = xr.DataArray(data, attrs={"units": units, "calendar": calendar})
    return xr.decode_cf(date.to_dataset(name="time"), use_cftime=True).time


def _datestr_to_cftime(date_str, calendar):
    # Again I feel this should be more elegant? For now I guess it works
    return _construct_cfdate([0], f"days since {date_str}", calendar)


def replace_time(
    ds, ref_date=None, ref_idx=0, freq="1MS", calendar=None, time_dim="time"
):
    """This function replaces the time encoding of a dataset acoording to `ref_date`.
    The ref date can be any index of ds.time (default is 0; meaning the first timestep of ds will be replaced with `ref_date`).
    """
    # ! I might be able to achieve some of this with time.shift
    # !

    if calendar is None:
        calendar = ds.time.encoding.get("calendar", "standard")

    if ref_date is None:
        ref_date = _maybe_unpack_date(ds.time[0])

    if isinstance(ref_date, str):
        ref_date = _maybe_unpack_date(_datestr_to_cftime(ref_date, calendar))

    # TODO: Check the frequency. Currently the logic only works on monthly intervals
    if freq != "1MS":
        raise ValueError("`replace_time` currently only works with monthly data.")

    # determine the start date
    # propagate the date back (this assumes stricly monthly data)

    year = _maybe_unpack_date(ref_date).year - (ref_idx // 12)
    month = _maybe_unpack_date(ref_date).month - (ref_idx % 12)

    if month <= 0:
        # move the year one more back
        year -= 1
        month = 12 + month

    attrs = ds.time.attrs

    start = f"{int(year):04d}-{int(month):02d}"

    ds = ds.assign_coords(
        time=xr.cftime_range(start, periods=len(ds.time), freq=freq, calendar=calendar)
    )
    ds.time.attrs = attrs
    return ds


def find_date_idx(time, date):
    """Finds the index of `date` within an array of cftime dates. This strictly requires monthly data.
    Might result in undesired behavior for other time frequencies.
    """
    # ! seems like I can refactor this with http://xarray.pydata.org/en/stable/generated/xarray.CFTimeIndex.get_loc.html#xarray.CFTimeIndex.get_loc

    date = _maybe_unpack_date(date)

    # easier approach: Find the difference in years and months
    year_diff = date.year - _maybe_unpack_date(time[0]).year
    month_diff = date.month - _maybe_unpack_date(time[0]).month

    return (year_diff * 12) + month_diff


def unify_time(parent, child, adjust_to="child"):
    """Uses the CMIP6 specific metadata (augmented by xmip....time_preprocessing!!!) to adjust parent time encoding to child experiment.
    Similar to `switch_to_child_time`, but sets the time parameters (e.g. calendar) explicitly to the child conventions
    """
    branch_time_in_parent = child.attrs.get("branch_time_in_parent")

    # if branch time is not in attrs do nothing
    if branch_time_in_parent is None:
        child_source_id = child.attrs.get("source_id", "not found")
        parent_source_id = parent.attrs.get("source_id", "not found")
        msg = (
            f"Could not unify time for [child:{child_source_id}|parent:{parent_source_id}]."
            "`branch_time_in_parent` not found in attributes."
        )
        warnings.warn(msg, UserWarning)
        return parent, child

    else:
        parent_calendar = parent.time.to_index().calendar
        child_calendar = child.time.to_index().calendar
        branch_time_parent = _construct_cfdate(
            child.attrs.get("branch_time_in_parent"),
            child.attrs.get("parent_time_units"),
            parent_calendar,
        )
        branch_time_child = _construct_cfdate(
            child.attrs.get("branch_time_in_child"),
            child.time.encoding.get("units"),
            child_calendar,
        )

        if adjust_to == "child":
            branch_idx_parent = find_date_idx(parent.time, branch_time_parent)
            return (
                replace_time(
                    parent,
                    branch_time_child,
                    ref_idx=branch_idx_parent,
                    calendar=child_calendar,
                ),
                child,
            )
        elif adjust_to == "parent":
            branch_idx_child = find_date_idx(child.time, branch_time_child)
            return parent, replace_time(
                child,
                branch_time_parent,
                ref_idx=branch_idx_child,
                calendar=parent_calendar,
            )
        else:
            raise ValueError(
                f"Input for `adjust_to` not valid. Got {adjust_to}. Expected either `child` or `parent`."
            )


def calculate_drift(
    reference, ds, variable, trend_years=250, compute_short_trends=False
):
    """Calculate the linear trend at every grid position for the given time (`trend_years`)
    starting from the date when `ds` was branched of from `ds_parent`.
    CMIP6 metadata must be present.

    Parameters
    ----------
    ds_parent : xr.Dataset
        The dataset from which the drift (trend) is calculated. Usually the preindustrial control run
    ds : xr.Dataset
        The dataset for which the drift is matched. This is usually the historical experiment.
        !For many models, each historical member is branched
    trend_years : int, optional
        The duration of the trend to compute in years, by default 250 (This is the lenght of
        historical+standard scenario, e.g. 1850-2100)
    """

    for attr in [
        "parent_variant_label",
        "parent_source_id",
        "branch_time_in_parent",
        "parent_time_units",
        "source_id",
        "variant_label",
    ]:
        if attr not in ds.attrs:
            raise ValueError(f"Could not find {attr} in attributes of `ds`.")

    # Check if the parent member id matches
    match_attrs = ["source_id", "variant_label"]
    for ma in match_attrs:
        if not ds.attrs[f"parent_{ma}"] in reference.attrs[ma]:
            raise ValueError(
                f'`ds_parent` {ma} ({reference.attrs[ma]}) not compatible with `ds` parent_{ma} ({ds.attrs[f"parent_{ma}"]})'
            )

    # find the branch date in the control run
    branch_time_reference = _construct_cfdate(
        ds.attrs["branch_time_in_parent"],
        ds.attrs["parent_time_units"],
        reference.time.to_index().calendar,
    )
    branch_idx_reference = find_date_idx(reference.time, branch_time_reference)
    # there might be some cases where this is not true. Figure out what to do when it happens.
    assert branch_idx_reference >= 0

    # cut the referenmce to the appropriate time frame
    reference_cut = reference.isel(
        time=slice(branch_idx_reference, branch_idx_reference + (12 * trend_years))
    )

    if len(reference_cut.time) == 0:
        raise RuntimeError(
            "Selecting from `reference` according to the branch time resulted in empty dataset. Check the metadata."
        )
        return None
    else:
        if len(reference_cut.time) < trend_years * 12:
            if compute_short_trends:
                warnings.warn(
                    f"reference dataset does not have the full {trend_years} years to calculate trend. Using {int(len(reference_cut.time)/12)} years only"
                )
            else:
                raise RuntimeError(
                    f"Reference dataset does not have the full {trend_years} years to calculate trend. Set `calculate_short_trend=True` to compute from a shorter timeseries"
                )

        time_range = xr.concat(
            [
                reference_cut.time[0].squeeze().drop_vars("time"),
                reference_cut.time[-1].squeeze().drop_vars("time"),
            ],
            dim="bnds",
        ).reset_coords(drop=True)

        # there is some problem when encoding very large years. for now ill preserve these only as
        # strings
        time_range = time_range.astype(str)

        # The polyfit implementation actually respects the units.
        # For now my implementation requires the slope to be in units .../month
        # I might be able to change this later and accomodate other time frequencies?
        # get rid of all the additional coords, which resets the time to an integer index

        reference_cut = reference_cut[variable]

        # TODO: This has pretty poor performance...need to find out why.
        # Reset time dimension to integer index.
        #         reference_cut = reference_cut.drop_vars("time")

        # linear regression slope is all we need here.
        #         reg = reference_cut.polyfit("time", 1).sel(degree=1).polyfit_coefficients

        reg_raw = linear_trend(
            reference_cut,
            "time",
        )

        # ! quite possibly the shittiest fix ever.
        # I changed the API over at xarrayutils and now I have to pay the price over here.
        # TODO: Might want to eliminate this ones the new xarrayutils version has matured.
        if xru.__version__ > "v0.1.3":
            reg = reg_raw.slope
        else:
            reg = reg_raw.sel(parameter="slope").drop_vars("parameter").squeeze()

        # again drop all the coordinates
        reg = reg.reset_coords(drop=True)

        reg = reg.to_dataset(name=variable)

        # add metadata about regression
        reg = reg.assign_coords(trend_time_range=time_range)
        reg.coords["trend_time_range"].attrs.update(
            {
                "standard_name": "regression_time_bounds",
                "long_name": "regression_time_in_reference_run",
            }
        )
        # reg should carry the attributes of `ds`
        # ? Maybe I should convert to a dataset?
        reg.attrs.update(ds.attrs)
        return reg


# TODO: I need a more generalized detrending? Based on indicies --> xarrayutils
# Then refactor this one here just for cmip6


def detrend_basic(da, da_slope, start_idx=0, dim="time", keep_attrs=True):
    """Basic detrending just based on time index, not date"""
    # now create a trend timeseries at each point
    # and the time indicies by the ref index. This way the trend is correctly calculated from the reference year.
    # this adapts the chunk structure from the input if its a dask array
    attrs = {k: v for k, v in da.attrs.items()}
    idx_start = -start_idx
    idx_stop = len(da.time) - start_idx
    if isinstance(da.data, dsa.Array):
        ref_time = da.isel({di: 0 for di in da.dims if di != dim})
        chunks = ref_time.chunks
        trend_time_idx_data = dsa.arange(
            idx_start, idx_stop, chunks=chunks, dtype=da.dtype
        )
    else:
        trend_time_idx_data = np.arange(idx_start, idx_stop, dtype=da.dtype)

    trend_time_idx = xr.DataArray(
        trend_time_idx_data,
        dims=[dim],
    )

    # chunk like the time dimension
    slope = da_slope.squeeze()

    trend = trend_time_idx * slope

    detrended = da - trend
    if keep_attrs:
        detrended.attrs.update(attrs)
    return detrended


def remove_trend(ds, ds_slope, variable, ref_date, check_mask=True):
    """Detrending method for cmip6 data. Only works with monthly data!
    This does not correct the time convention. Be careful with experiements that have
    a non compatible time convention (often control runs.)
    """

    if not isinstance(ds, xr.Dataset):
        raise ValueError("`ds` input needs to be a dataset")

    if not isinstance(ds_slope, xr.Dataset):
        raise ValueError("`ds_slope` input needs to be a dataset")

    da = ds[variable]
    da_slope = ds_slope[variable]

    da, da_slope = xr.align(da, da_slope, join="override")

    if check_mask:
        nanmask_data = np.isnan(da.isel(time=[0, len(da.time) // 2, -1])).all("time")
        nanmask_slope = np.isnan(da_slope)
        # perform a quick test to see if the land is aligned properly
        if np.logical_xor(nanmask_data, nanmask_slope).any():
            raise ValueError(
                "Nanmask between data and slope array not identical. Check input and disable `check_mask` to skip this test"
            )

    ref_calendar = da.time.to_index().calendar
    ref_date = xr.cftime_range(ref_date, periods=1, calendar=ref_calendar)

    # Find the index corresponding to the ref date (this can be outside the range of the actual data)
    ref_idx = find_date_idx(da.time, ref_date)

    detrended = detrend_basic(
        da, da_slope, start_idx=ref_idx, dim="time", keep_attrs=True
    )

    # add information to track which data was used to remove trend
    if "trend_time_range" in ds_slope.coords:
        trend_start = ds_slope.trend_time_range.isel(bnds=0).load().data.tolist()
        trend_stop = ds_slope.trend_time_range.isel(bnds=1).load().data.tolist()

    else:
        trend_start = "not-available"
        trend_stop = "not-available"
        warnings.warn(
            "`ds_slope` did not have information about the time over which the slope was calculated. Check the input."
        )

    detrended.attrs[
        "drift_removed"
    ] = f"linear_trend_{cmip6_dataset_id(ds_slope)}_{trend_start}_{trend_stop}"

    return detrended


def match_and_remove_trend(
    ddict, trend_ddict, ref_date="1850", nomatch="warn", **detrend_kwargs
):
    """Find and remove trend files from a dictonary of datasets

    Parameters
    ----------
    ddict : dict
        dictionary with xr.Datasets which should get a trend/drift removed
    trend_ddict : dict
        dictionary with results of linear regressions. These should be removed from the datasets in `ddict`
    ref_date : str, optional
        Start date of the trend, by default "1850"
    nomatch : str, optional
        Define the behavior when for a given dataset in `ddict` there is no matching trend dataset in `trend_ddict`.
        Can be `warn`, `raise`, or `ignore`, by default 'warn'

    Returns
    -------
    dict
        Dictionary of detrended dataasets. Only contains values of `ddict` that actually had a trend removed.

    """
    ddict_detrended = {}
    match_attrs = [ma for ma in EXACT_ATTRS if ma not in ["experiment_id"]] + [
        "variable_id"
    ]

    for k, ds in ddict.items():
        trend_ds = _match_datasets(
            ds, trend_ddict, match_attrs, pop=False, unique=True, nomatch=nomatch
        )
        if len(trend_ds) == 2:
            trend_ds = trend_ds[
                1
            ]  # this is a bit clunky. _match_datasest does return the input ds, so we have to grab the second one?
            # I guess I could pass *trend_ds, but that is not very readable
            variable = ds.attrs["variable_id"]
            da_detrended = ds.assign(
                {
                    variable: remove_trend(
                        ds, trend_ds, variable, ref_date=ref_date, **detrend_kwargs
                    )
                }
            )
            # should this just return a dataset instead?
            ddict_detrended[k] = da_detrended

    return ddict_detrended
