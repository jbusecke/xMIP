import random
import string

import numpy as np
import pytest
import xarray as xr

from cmip6_preprocessing.postprocessing import (
    _parse_metric,
    combine_datasets,
    concat_members,
    match_metrics,
    merge_variables,
)


@pytest.fixture(params=["metric", "other"])
def metricname(request):
    return request.param


def random_ds(time_coords=False, attrs={}):
    """Create random dataset"""
    nx, ny, nz, nt = (3, 2, 4, 6)
    data = np.random.rand(nx, ny, nz, nt)
    da = xr.DataArray(data, dims=["x", "y", "z", "time"])
    if time_coords:
        da = da.assign_coords(time=xr.cftime_range("2000", periods=nt))
    ds = da.to_dataset(name="data")
    ds.attrs = attrs
    return ds


def test_parse_metric(metricname):
    # create a dataset
    ds = random_ds()
    # create a metric dataset
    ds_metric = random_ds().isel(z=0, time=0).rename({"data": metricname})
    metric = ds_metric[metricname]
    metric.attrs.update({"check": "carry"})

    # parse the metric
    ds_parsed = _parse_metric(ds, metric)
    assert metricname in ds_parsed.coords
    xr.testing.assert_allclose(ds_parsed[metricname].reset_coords(drop=True), metric)
    assert (
        ds_parsed[metricname].attrs["parsed_with"]
        == "cmip6_preprocessing/postprocessing/_parse_metric"
    )
    # check that existing attrs are conserved
    assert ds_parsed[metricname].attrs["check"] == "carry"


def test_parse_metric_exceptions(metricname):
    # create a dataset
    ds = random_ds()
    # create a metric dataset
    ds_metric = random_ds().isel(z=0, time=0).rename({"data": metricname})

    # provide dataset instead of dataarray
    with pytest.raises(ValueError):
        ds_parsed = _parse_metric(ds, ds_metric)


def test_parse_metric_exceptions_input_name():
    # create a dataset
    ds = random_ds()
    # create a metric dataset
    ds_metric = random_ds().isel(z=0, time=0)
    # set attributes
    ds.attrs = {"activity_id": "a"}

    # provide dataarray without name
    with pytest.warns(RuntimeWarning) as warninfo:
        da_metric_nameless = ds_metric["data"]
        da_metric_nameless.name = None

        ds_parsed = _parse_metric(ds, da_metric_nameless)
    assert (
        warninfo[0].message.args[0]
        == "a.none.none.none.none.none.none:`metric` has no name. This might lead to problems down the line."
    )


def test_parse_metric_exception_dim_length():
    metricname = "metric"
    # create a dataset
    ds = random_ds()
    # create a metric dataset
    ds_metric = random_ds().isel(z=0, time=0).rename({"data": metricname})
    # set attributes
    ds.attrs = {"activity_id": "a", "grid_label": "g"}

    # provide dataarray with non-matching dimensions
    with pytest.raises(ValueError) as execinfo:
        ds_parsed = _parse_metric(
            ds, ds_metric.isel(x=slice(0, -1), y=slice(1, None))[metricname]
        )
    msg = "a.none.none.none.none.g.none:`metric` dimensions ['x:2', 'y:1'] do not match `ds` ['x:3', 'y:2']."
    assert execinfo.value.args[0] == msg


def test_parse_metric_dim_alignment():
    metricname = "metric"
    # create a dataset
    ds = random_ds(time_coords=True)
    # create a metric dataset
    ds_metric = (
        random_ds(time_coords=True)
        .isel(z=0, time=slice(0, -1))
        .rename({"data": metricname})
    )
    # set attributes
    ds.attrs = {"activity_id": "a", "grid_label": "g"}

    print(ds)

    # Allow alignment
    with pytest.warns(UserWarning) as warninfo:
        ds_parsed = _parse_metric(
            ds, ds_metric[metricname], dim_length_conflict="align"
        )

    xr.testing.assert_allclose(ds_parsed.time, ds_metric.time)

    msg = "a.none.none.none.none.g.none:`metric` dimensions ['time:5'] do not match `ds` ['time:6']. Aligning the data on `inner`"
    assert warninfo[0].message.args[0] == msg


def test_match_metrics(metricname):
    # create a few different datasets
    ds_a = random_ds()
    ds_b = random_ds()
    ds_c = random_ds()
    ds_d = random_ds()
    ds_e = random_ds()

    # Give them cmip attrs
    ds_a.attrs = {
        "source_id": "a",
        "grid_label": "a",
        "experiment_id": "a",
        "table_id": "a",
        "variant_label": "a",
        "version": "a",
    }
    ds_b.attrs = {
        "source_id": "a",
        "grid_label": "a",
        "experiment_id": "a",
        "table_id": "b",
        "variant_label": "b",
        "version": "b",
    }
    ds_c.attrs = {
        "source_id": "a",
        "grid_label": "a",
        "experiment_id": "b",
        "table_id": "b",
        "variant_label": "a",
        "version": "b",
    }
    ds_d.attrs = {
        "source_id": "a",
        "grid_label": "b",
        "experiment_id": "a",
        "table_id": "a",
        "variant_label": "a",
        "version": "a",
    }
    ds_e.attrs = {
        "source_id": "b",
        "grid_label": "a",
        "experiment_id": "a",
        "table_id": "a",
        "variant_label": "a",
        "version": "a",
    }

    # now create a metric (which does not vary in time) which matches ds_a
    ds_metric = random_ds().isel(time=0).rename({"data": metricname})
    ds_metric.attrs = ds_a.attrs

    def _assert_parsed_ds_dict(ddict_parsed, expected, match_keys, strict=True):
        expected = expected.copy()
        for i in match_keys:
            ds_parsed = ddict_parsed[i]
            assert metricname in list(ds_parsed.variables)
            xr.testing.assert_allclose(
                ds_parsed[metricname].reset_coords(drop=True), expected
            )
        if strict:
            for i in [ii for ii in ddict_parsed.keys() if ii not in match_keys]:
                ds_parsed = ddict_parsed[i]
                assert metricname not in ds_parsed.variables

    # With the default options I expect that this gets parsed into a,b,c (all the same source_id and grid_label)
    # but not d and e
    ds_dict = {"a": ds_a, "b": ds_b, "c": ds_c, "d": ds_d, "e": ds_e}
    metric_dict = {"something": ds_metric}
    expected = ds_metric[metricname]

    ds_dict_parsed = match_metrics(ds_dict, metric_dict, [metricname])
    _assert_parsed_ds_dict(ds_dict_parsed, ds_metric[metricname], ["a", "b", "c"])

    # Now change the matching parameter
    ds_dict_parsed = match_metrics(
        ds_dict, metric_dict, match_variables=[metricname], match_attrs="exact"
    )
    _assert_parsed_ds_dict(ds_dict_parsed, expected, ["a"])

    ds_dict_parsed = match_metrics(
        ds_dict,
        metric_dict,
        [metricname],
        match_attrs=["source_id", "grid_label", "experiment_id"],
    )
    _assert_parsed_ds_dict(ds_dict_parsed, expected, ["a", "b"])

    ds_dict_parsed = match_metrics(
        ds_dict,
        metric_dict,
        [metricname],
        match_attrs=["source_id", "grid_label", "variant_label"],
    )
    _assert_parsed_ds_dict(ds_dict_parsed, expected, ["a", "c"])

    # Now give the metric the attributes of e and check
    ds_metric.attrs = ds_e.attrs

    ds_dict = {"a": ds_a, "b": ds_b, "c": ds_c, "d": ds_d, "e": ds_e}
    metric_dict = {"something": ds_metric}

    ds_dict_parsed = match_metrics(ds_dict, metric_dict, match_variables=[metricname])
    _assert_parsed_ds_dict(ds_dict_parsed, ds_metric[metricname], ["e"])

    # Check that a metric with time dimension is never parsed anywhere, except an exact match
    ds_metric = random_ds().rename({"data": metricname})
    ds_metric.attrs = ds_a.attrs

    ds_dict = {"a": ds_a, "b": ds_b, "c": ds_c, "d": ds_d, "e": ds_e}
    metric_dict = {"something": ds_metric}

    ds_dict_parsed = match_metrics(ds_dict, metric_dict, match_variables=[metricname])
    _assert_parsed_ds_dict(ds_dict_parsed, ds_metric[metricname], ["a"])


def test_match_metrics_closer(metricname):
    # Test to see if a metric dataset with more matching attrs is preferred.

    ds_a = random_ds()
    ds_b = random_ds()
    ds_c = random_ds()

    # Give them cmip attrs
    ds_a.attrs = {
        "source_id": "a",
        "grid_label": "a",
        "experiment_id": "a",
        "table_id": "a",
        "variant_label": "a",
        "version": "a",
    }
    ds_b.attrs = {
        "source_id": "a",
        "grid_label": "a",
        "experiment_id": "a",
        "table_id": "b",
        "variant_label": "b",
        "version": "b",
    }
    ds_c.attrs = {
        "source_id": "a",
        "grid_label": "a",
        "experiment_id": "b",
        "table_id": "b",
        "variant_label": "a",
        "version": "b",
    }

    ds_metric_a = random_ds().isel(time=0).rename({"data": metricname})
    ds_metric_a.attrs = ds_a.attrs
    ds_metric_c = random_ds().isel(time=0).rename({"data": metricname})
    ds_metric_c.attrs = ds_c.attrs

    ds_dict = {"c": ds_c}
    metric_dict = {
        "exact_c": ds_metric_c,
        "exact_a": ds_metric_a,
    }
    ds_dict_parsed = match_metrics(ds_dict, metric_dict, match_variables=[metricname])
    xr.testing.assert_allclose(
        ds_dict_parsed["c"][metricname].reset_coords(drop=True), ds_metric_c[metricname]
    )
    assert ds_dict_parsed["c"][metricname].attrs["original_key"] == "exact_c"


def test_match_metrics_exceptions(metricname):
    # give a dataset that has member_id as dim (indicator that it was aggregated).

    attrs = {
        "source_id": "a",
        "grid_label": "a",
        "experiment_id": "a",
        "table_id": "a",
        "variant_label": "a",
        "version": "a",
    }
    ds = random_ds().rename({"z": "member_id"})
    ds.attrs = attrs
    ds_metric = random_ds().rename({"data": metricname})
    ds_metric.attrs = attrs
    with pytest.raises(ValueError):
        match_metrics({"a": ds}, {"aa": ds_metric}, [metricname])


def test_match_metrics_align_dims():
    metricname = "metric"
    attrs = {
        "source_id": "a",
        "grid_label": "a",
        "experiment_id": "a",
        "table_id": "a",
        "variant_label": "a",
        "version": "a",
    }
    ds = random_ds(time_coords=True)
    ds.attrs = attrs
    ds_metric = (
        random_ds(time_coords=True).isel(time=slice(0, -1)).rename({"data": metricname})
    )
    ds_metric.attrs = attrs
    with pytest.warns(UserWarning) as warninfo:
        ddict_matched = match_metrics(
            {"a": ds},
            {"aa": ds_metric},
            [metricname],
            print_statistics=True,
            dim_length_conflict="align",
        )
    msg = "none.none.a.a.a.a.a:`metric` dimensions ['time:5'] do not match `ds` ['time:6']. Aligning the data on `inner`"
    assert warninfo[0].message.args[0] == msg

    xr.testing.assert_allclose(ddict_matched["a"].time, ds_metric.time)


def test_match_metrics_print_statistics(capsys, metricname):
    # give a dataset that has member_id as dim (indicator that it was aggregated).

    attrs = {
        "source_id": "a",
        "grid_label": "a",
        "experiment_id": "a",
        "table_id": "a",
        "variant_label": "a",
        "version": "a",
    }
    ds = random_ds()
    ds.attrs = attrs
    ds_metric = random_ds().rename({"data": metricname})
    ds_metric.attrs = attrs

    match_metrics({"a": ds}, {"aa": ds_metric}, [metricname], print_statistics=True)

    captured = capsys.readouterr()

    assert "Processed 1 datasets." in captured.out
    assert "Exact matches:" + str({metricname: 1}) in captured.out
    assert "Other matches:" + str({metricname: 0}) in captured.out
    assert "No match found:" + str({metricname: 0}) in captured.out


@pytest.mark.parametrize("combine_func_kwargs", [{}, {"compat": "override"}])
def test_combine_datasets_merge(combine_func_kwargs):
    attrs_a = {
        "source_id": "a",
        "grid_label": "a",
        "experiment_id": "a",
        "table_id": "a",
        "variant_label": "a",
        "version": "a",
    }

    attrs_b = {
        "source_id": "b",
        "grid_label": "b",
        "experiment_id": "b",
        "table_id": "b",
        "variant_label": "b",
        "version": "b",
    }

    # Create some datasets with a/b attrs
    ds_a_temp = random_ds(attrs=attrs_a).rename({"data": "temp"})
    ds_a_salt = random_ds(attrs=attrs_a).rename({"data": "salt"})

    ds_b_temp = random_ds(attrs=attrs_b).rename({"data": "temp"})
    ds_b_salt = random_ds(attrs=attrs_b).rename({"data": "salt"})
    ds_b_other = random_ds(attrs=attrs_b).rename({"data": "other"})

    ds_dict = {
        "".join(random.choices(string.ascii_letters, k=4)): ds
        for ds in [ds_a_salt, ds_a_temp, ds_b_salt, ds_b_temp, ds_b_other]
    }

    # Group together the expected 'matches'
    expected = {
        "a.a.a.a.a": [ds_a_salt, ds_a_temp],
        "b.b.b.b.b": [ds_b_other, ds_b_temp, ds_b_salt],
    }

    result = combine_datasets(
        ds_dict,
        xr.merge,
        combine_func_kwargs=combine_func_kwargs,
    )
    for k in expected.keys():
        assert k in list(result.keys())
        xr.testing.assert_equal(result[k], xr.merge(expected[k], **combine_func_kwargs))


def test_merge_variables():
    attrs_a = {
        "source_id": "a",
        "grid_label": "a",
        "experiment_id": "a",
        "table_id": "a",
        "variant_label": "a",
        "version": "a",
    }

    attrs_b = {
        "source_id": "b",
        "grid_label": "b",
        "experiment_id": "b",
        "table_id": "b",
        "variant_label": "b",
        "version": "b",
    }

    # Create some datasets with a/b attrs
    ds_a_temp = random_ds(attrs=attrs_a).rename({"data": "temp"})
    ds_a_salt = random_ds(attrs=attrs_a).rename({"data": "salt"})

    ds_b_temp = random_ds(attrs=attrs_b).rename({"data": "temp"})
    ds_b_salt = random_ds(attrs=attrs_b).rename({"data": "salt"})
    ds_b_other = random_ds(attrs=attrs_b).rename({"data": "other"})

    ds_dict = {
        "".join(random.choices(string.ascii_letters, k=4)): ds
        for ds in [ds_a_salt, ds_a_temp, ds_b_salt, ds_b_temp, ds_b_other]
    }

    # Group together the expected 'matches'
    expected = {
        "a.a.a.a.a": [ds_a_salt, ds_a_temp],
        "b.b.b.b.b": [ds_b_other, ds_b_temp, ds_b_salt],
    }

    result = merge_variables(ds_dict)

    for k in expected.keys():
        assert k in list(result.keys())
        xr.testing.assert_equal(
            result[k],
            xr.merge(
                expected[k],
                **{
                    "compat": "override",
                    "join": "exact",
                    "combine_attrs": "drop_conflicts",
                },
            ),
        )


@pytest.mark.parametrize("concat_kwargs", [{}, {"compat": "override"}])
def test_concat_members(concat_kwargs):
    concat_kwargs = {}

    attrs_a = {
        "source_id": "a",
        "grid_label": "a",
        "experiment_id": "a",
        "table_id": "a",
        "variant_label": "a",
        "version": "a",
    }

    attrs_b = {k: v for k, v in attrs_a.items()}
    attrs_b["variant_label"] = "b"

    attrs_c = {k: v for k, v in attrs_b.items()}
    attrs_c["source_id"] = "c"

    # Create some datasets with a/b attrs
    ds_a_temp = random_ds(attrs=attrs_a).rename({"data": "temp"})
    ds_b_temp = random_ds(attrs=attrs_b).rename({"data": "temp"})

    ds_c_other = random_ds(attrs=attrs_c).rename({"data": "other"})

    ds_dict = {
        "".join(random.choices(string.ascii_letters, k=4)): ds
        for ds in [ds_a_temp, ds_b_temp, ds_c_other]
    }

    # Group together the expected 'matches'
    expected = {
        "a.a.a.a": [ds_a_temp, ds_b_temp],
        "c.a.a.a": [ds_c_other],
    }

    result = concat_members(
        ds_dict,
        concat_kwargs=concat_kwargs,
    )
    for k in expected.keys():
        assert k in list(result.keys())
        xr.testing.assert_equal(
            result[k], xr.concat(expected[k], "member_id", **concat_kwargs)
        )


@pytest.mark.parametrize("func", [merge_variables, concat_members])
def test_aggregate_error(func):
    # check if an error is raised if a dataset with `member_id` dimension is passed.

    ds = xr.concat([random_ds(), random_ds()], "member_id")
    with pytest.raises(ValueError):
        func({"test": ds})
