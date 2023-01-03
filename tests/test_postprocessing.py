import random
import string

import numpy as np
import pytest
import xarray as xr
import xesmf

from xmip.postprocessing import (
    _construct_and_promote_member_id,
    _parse_metric,
    combine_datasets,
    concat_experiments,
    concat_members,
    interpolate_grid_label,
    match_metrics,
    merge_variables,
    pick_first_member,
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
        == "xmip/postprocessing/_parse_metric"
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
        _parse_metric(ds, ds_metric)


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

        _parse_metric(ds, da_metric_nameless)
    assert (
        warninfo[0].message.args[0]
        == "a.none.none.none.none.none.none.none.none:`metric` has no name. This might lead to problems down the line."
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
        _parse_metric(ds, ds_metric.isel(x=slice(0, -1), y=slice(1, None))[metricname])
    msg = "a.none.none.none.none.none.g.none.none:`metric` dimensions ['x:2', 'y:1'] do not match `ds` ['x:3', 'y:2']."
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

    msg = "a.none.none.none.none.none.g.none.none:`metric` dimensions ['time:5'] do not match `ds` ['time:6']. Aligning the data on `inner`"
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

    # FIXME: This needs to be factored out
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


def test_match_metrics_missing_non_match_attr():
    """This test ensures that as long as the provided `match_metrics` are
    given they will be matched. This is relevant if e.g. the variant label
    has been removed due to merging"""
    metricname = "area"
    ds_a = random_ds()
    ds_a.attrs = {
        "source_id": "a",
        "grid_label": "a",
    }
    ds_metric = random_ds().isel(time=0).rename({"data": metricname})
    ds_metric.attrs = ds_a.attrs

    ds_dict = {"a": ds_a}
    metric_dict = {"something": ds_metric}
    expected = ds_metric[metricname]

    ds_dict_parsed = match_metrics(ds_dict, metric_dict, [metricname])

    assert "a" in ds_dict_parsed.keys()
    # TODO this should be factored out into _assert_parsed_ds_dict from the test above
    xr.testing.assert_allclose(
        expected, ds_dict_parsed["a"][metricname].reset_coords(drop=True)
    )


def test_match_metrics_missing_match_attrs():
    """If one of the `match_attrs` is not in the dataset this should error out"""
    metricname = "area"
    ds_a = random_ds()
    ds_a.attrs = {
        "source_id": "a",
    }
    ds_metric = random_ds().isel(time=0).rename({"data": metricname})
    ds_metric.attrs = ds_a.attrs

    ds_dict = {"a": ds_a}
    metric_dict = {"something": ds_metric}
    with pytest.raises(
        ValueError,
        match="Cannot match datasets because at least one of the datasets does not contain all attributes",
    ):
        match_metrics(ds_dict, metric_dict, [metricname])


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


def test_match_metrics_align_dims():
    metricname = "metric"
    attrs = {
        "source_id": "a",
        "grid_label": "a",
        "experiment_id": "a",
        "table_id": "a",
        "variant_label": "a",
        "version": "a",
        "variable_id": "a",
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
    msg = "none.none.a.a.a.a.a.a.a:`metric` dimensions ['time:5'] do not match `ds` ['time:6']. Aligning the data on `inner`"
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


def test_match_metrics_match_variable_str_input():
    # give a dataset that has member_id as dim (indicator that it was aggregated).
    metricname = "area"
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

    ds_dict_parsed_list = match_metrics({"a": ds}, {"aa": ds_metric}, [metricname])
    ds_dict_parsed_str = match_metrics({"a": ds}, {"aa": ds_metric}, metricname)
    xr.testing.assert_equal(ds_dict_parsed_str["a"], ds_dict_parsed_list["a"])


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
    # promote the member_id like in concat_members
    expected = {
        "a.a.a.a": [
            _construct_and_promote_member_id(ds_a_temp),
            _construct_and_promote_member_id(ds_b_temp),
        ],
        "c.a.a.a": [_construct_and_promote_member_id(ds_c_other)],
    }

    result = concat_members(
        ds_dict,
        concat_kwargs=concat_kwargs,
    )
    for k in expected.keys():
        assert k in list(result.keys())
        xr.testing.assert_equal(
            result[k],
            xr.concat(expected[k], "member_id", **concat_kwargs),
        )
    # assert that member_id is a proper coordinate
    assert "member_id" in result["a.a.a.a"].coords
    for member in ["a", "b"]:
        assert member in result["a.a.a.a"].member_id


def test_concat_members_existing_member_dim():
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

    # Create some datasets with a/b attrs
    ds_a = random_ds(attrs=attrs_a).rename({"data": "temp"})
    ds_b = random_ds(attrs=attrs_b).rename({"data": "temp"})

    ds_a_promoted = ds_a.expand_dims({"member_id": [ds_a.attrs["variant_label"]]})
    ds_b_promoted = ds_b.expand_dims({"member_id": [ds_b.attrs["variant_label"]]})

    # testing mixed case
    ds_dict = {"some": ds_a_promoted, "thing": ds_b}

    # promote the member_id like in concat_members
    expected = xr.concat([ds_a_promoted, ds_b_promoted], "member_id")

    result = concat_members(
        ds_dict,
    )

    xr.testing.assert_equal(
        result["a.a.a.a"],
        expected,
    )


def test_concat_members_existing_member_dim_different_warning():
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

    # Create some datasets with a/b attrs
    ds_a = random_ds(attrs=attrs_a).rename({"data": "temp"})
    ds_b = random_ds(attrs=attrs_b).rename({"data": "temp"})

    ds_a_promoted_wrong = ds_a.expand_dims({"member_id": ["something"]})

    # testing mixed case
    ds_dict = {"some": ds_a_promoted_wrong, "thing": ds_b}
    msg = "but this is different from the reconstructed value"
    # TODO: Had trouble here when putting in the actual values I expected.
    # Probably some regex shit. This should be enough for now
    with pytest.warns(UserWarning, match=msg):
        concat_members(
            ds_dict,
        )


def test_concat_members_reconstruct_from_sub_experiment_id():
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
    attrs_b["sub_experiment_id"] = "sub_something"

    # Create some datasets with a/b attrs
    ds_a = random_ds(attrs=attrs_a).rename({"data": "temp"})
    ds_b = random_ds(attrs=attrs_b).rename({"data": "temp"})

    ds_a_promoted = ds_a.expand_dims({"member_id": ["a"]})
    ds_b_promoted = ds_b.expand_dims({"member_id": ["sub_something-b"]})

    # testing mixed case
    ds_dict = {"some": ds_a, "thing": ds_b}

    # promote the member_id like in concat_members
    expected = xr.concat([ds_a_promoted, ds_b_promoted], "member_id")

    result = concat_members(
        ds_dict,
    )

    xr.testing.assert_equal(
        result["a.a.a.a"],
        expected,
    )


@pytest.mark.parametrize("concat_kwargs", [{}, {"compat": "override"}])
def test_concat_experiments(concat_kwargs):
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
    attrs_b["experiment_id"] = "b"

    attrs_c = {k: v for k, v in attrs_b.items()}
    attrs_c["source_id"] = "c"

    # Create some datasets with a/b attrs
    ds_a_temp = random_ds(attrs=attrs_a).rename({"data": "temp"})
    ds_a_temp = ds_a_temp.assign_coords(
        time=xr.cftime_range("2000", periods=len(ds_a_temp.time))
    )

    ds_b_temp = random_ds(attrs=attrs_b).rename({"data": "temp"})
    ds_b_temp = ds_b_temp.assign_coords(
        time=xr.cftime_range("1850", periods=len(ds_b_temp.time))
    )

    ds_c_other = random_ds(attrs=attrs_c).rename({"data": "other"})
    ds_c_other = ds_c_other.assign_coords(
        time=xr.cftime_range("2000", periods=len(ds_c_other.time))
    )

    ds_dict = {
        "".join(random.choices(string.ascii_letters, k=4)): ds
        for ds in [ds_a_temp, ds_b_temp, ds_c_other]
    }

    # Group together the expected 'matches'
    expected = {
        "a.a.a.a": [ds_b_temp, ds_a_temp],  # these should be sorted in this order!
        "c.a.a.a": [ds_c_other],
    }

    result = concat_experiments(
        ds_dict,
        concat_kwargs=concat_kwargs,
    )
    for k in expected.keys():
        assert k in list(result.keys())
        xr.testing.assert_equal(
            result[k], xr.concat(expected[k], "time", **concat_kwargs)
        )

    # shuffle the dict entries (not sure that is actually then processed in that order...)
    ds_dict = {
        "".join(random.choices(string.ascii_letters, k=4)): ds
        for ds in [ds_b_temp, ds_c_other, ds_a_temp]
    }

    # Group together the expected 'matches'
    expected = {
        "a.a.a.a": [ds_b_temp, ds_a_temp],  # these should be sorted in this order!
        "c.a.a.a": [ds_c_other],
    }

    result = concat_experiments(
        ds_dict,
        concat_kwargs=concat_kwargs,
    )
    for k in expected.keys():
        assert k in list(result.keys())
        xr.testing.assert_equal(
            result[k], xr.concat(expected[k], "time", **concat_kwargs)
        )


def test_pick_first_member():

    attrs_a = {
        "source_id": "a",
        "grid_label": "a",
        "experiment_id": "a",
        "table_id": "a",
        "variant_label": "a1b2",  # TODO: I might have to reevaluate this for some of the damip experiments (the only ones where variant_label!=member_id)
        "version": "a",
        "variable_id": "a",
    }

    attrs_b = {k: v for k, v in attrs_a.items()}
    attrs_b["variant_label"] = "a1b1"

    attrs_c = {k: v for k, v in attrs_b.items()}
    attrs_c["source_id"] = "b"
    attrs_c["variant_label"] = "a1b1"

    attrs_d = {k: v for k, v in attrs_b.items()}
    attrs_d["source_id"] = "b"
    attrs_d["variant_label"] = "a2b1"

    # Create some datasets with a/b attrs
    ds_a = random_ds(attrs=attrs_a).rename({"data": "temp"})
    ds_b = random_ds(attrs=attrs_b).rename({"data": "temp"})
    ds_c = random_ds(attrs=attrs_c).rename({"data": "temp"})
    ds_d = random_ds(attrs=attrs_d).rename({"data": "temp"})

    ds_dict = {
        "".join(random.choices(string.ascii_letters, k=4)): ds
        for ds in [ds_a, ds_b, ds_c, ds_d]
    }

    # Group together the expected 'matches'
    expected = {
        "a.a.a.a.a": ds_b,
        "b.a.a.a.a": ds_c,
    }

    result = pick_first_member(
        ds_dict,
    )
    print(result)
    print(result.keys())
    for k in expected.keys():
        assert k in list(result.keys())
        xr.testing.assert_equal(result[k], expected[k])


@pytest.mark.parametrize("verbose", [True, False])
def test_interpolate_grid_label(verbose):
    # build three datasets. Basically both datasets are available on grid_label hr (high res), but only one on grid_label lr (low res)
    ds_lr_vara = xr.DataArray(
        np.random.rand(4, 5, 6),
        dims=["x", "y", "lev"],
        coords={
            "lon": ("x", np.linspace(0, 360, 4)),
            "lat": ("y", np.linspace(-90, 90, 5)),
        },
    ).to_dataset(name="vara")
    ds_lr_vara.attrs = {
        "source_id": "a",
        "grid_label": "lr",
        "experiment_id": "a",
        "table_id": "a",
        "variant_label": "a",
        "version": "a",
        "variable_id": "vara",
    }

    ds_hr_vara = xr.DataArray(
        np.random.rand(10, 50, 6),
        dims=["x", "y", "lev"],
        coords={
            "lon": ("x", np.linspace(0, 360, 10)),
            "lat": (["y"], np.linspace(-90, 90, 50)),
        },
    ).to_dataset(name="vara")

    ds_hr_vara.attrs = {
        "source_id": "a",
        "grid_label": "hr",
        "experiment_id": "a",
        "table_id": "a",
        "variant_label": "a",
        "version": "a",
        "variable_id": "vara",
    }

    ds_hr_varb = xr.DataArray(
        np.random.rand(10, 50, 6),
        dims=["x", "y", "lev"],
        coords={
            "lon": ("x", np.linspace(0, 360, 10)),
            "lat": (["y"], np.linspace(-90, 90, 50)),
        },
    ).to_dataset(name="varb")

    ds_hr_varb.attrs = {
        "source_id": "a",
        "grid_label": "hr",
        "experiment_id": "a",
        "table_id": "a",
        "variant_label": "a",
        "version": "a",
        "variable_id": "varb",
    }

    # put in a dataset that has consistent grid_label

    ds_simple_vara = xr.DataArray(
        np.random.rand(10, 50, 6),
        dims=["x", "y", "lev"],
        coords={
            "lon": ("x", np.linspace(0, 360, 10)),
            "lat": (["y"], np.linspace(-90, 90, 50)),
        },
    ).to_dataset(name="vara")

    ds_simple_vara.attrs = {
        "source_id": "b",
        "grid_label": "hr",
        "experiment_id": "a",
        "table_id": "a",
        "variant_label": "a",
        "version": "a",
        "variable_id": "vara",
    }

    ds_simple_varb = xr.DataArray(
        np.random.rand(10, 50, 6),
        dims=["x", "y", "lev"],
        coords={
            "lon": ("x", np.linspace(0, 360, 10)),
            "lat": (["y"], np.linspace(-90, 90, 50)),
        },
    ).to_dataset(name="varb")

    ds_simple_varb.attrs = {
        "source_id": "b",
        "grid_label": "hr",
        "experiment_id": "a",
        "table_id": "a",
        "variant_label": "a",
        "version": "a",
        "variable_id": "varb",
    }

    # put in another rando dataset

    ds_rando = xr.DataArray(
        np.random.rand(14, 55, 6),
        dims=["x", "y", "lev"],
        coords={
            "lon": ("x", np.linspace(0, 360, 14)),
            "lat": (["y"], np.linspace(-90, 90, 55)),
        },
    ).to_dataset(name="varb")

    ds_rando.attrs = {
        "source_id": "b",
        "grid_label": "blubb",
        "experiment_id": "a",
        "table_id": "a",
        "variant_label": "a",
        "version": "a",
        "variable_id": "varb",
    }

    ddict = {
        "a_lr": ds_lr_vara,
        "a_hr": ds_hr_vara,
        "b_hr": ds_hr_varb,
        "a_simple": ds_simple_vara,
        "b_simple": ds_simple_varb,
        "rando": ds_rando,
    }

    expected_simple = xr.merge([ds_simple_vara, ds_simple_varb])

    # Prefer the high res version (no interpolation needed)
    combined_ddict = interpolate_grid_label(
        ddict, target_grid_label="hr", verbose=verbose
    )

    expected = xr.merge([ds_hr_vara, ds_hr_varb], combine_attrs="drop_conflicts")

    xr.testing.assert_allclose(combined_ddict["a.a.a.a"], expected)
    xr.testing.assert_allclose(combined_ddict["b.a.a.a"], expected_simple)

    # now the other way around (interpolation needed for variable whatever_else)
    combined_ddict = interpolate_grid_label(ddict, target_grid_label="lr")

    regridder = xesmf.Regridder(
        ds_hr_varb, ds_lr_vara, "bilinear", periodic=True, ignore_degenerate=True
    )

    regridded = regridder(ds_hr_varb)
    expected = xr.merge([ds_lr_vara, regridded])

    xr.testing.assert_allclose(combined_ddict["a.a.a.a"], expected)
    xr.testing.assert_allclose(combined_ddict["b.a.a.a"], expected_simple)

    assert combined_ddict["a.a.a.a"].varb.attrs["xmip_regrid_method"] == "bilinear"


def test_nested_operations():
    # set up a few datasets to combine
    ds_1 = xr.Dataset(
        {"a": ("x", np.array([1]))},
        attrs={
            "source_id": "a",
            "variant_label": "a",
            "variable_id": "a",
            "experiment_id": "a",
        },
    )
    ds_2 = xr.Dataset(
        {"a": ("x", np.array([2]))},
        attrs={
            "source_id": "a",
            "variant_label": "b",
            "variable_id": "a",
            "experiment_id": "a",
        },
    )
    ds_3 = xr.Dataset(
        {"b": ("x", np.array([3]))},
        attrs={
            "source_id": "a",
            "variant_label": "a",
            "variable_id": "b",
            "experiment_id": "a",
        },
    )
    ds_4 = xr.Dataset(
        {"b": ("x", np.array([4]))},
        attrs={
            "source_id": "a",
            "variant_label": "b",
            "variable_id": "b",
            "experiment_id": "a",
        },
    )
    ddict = {"ds1": ds_1, "ds2": ds_2, "ds3": ds_3, "ds4": ds_4}

    ddict = {k: _construct_and_promote_member_id(ds) for k, ds in ddict.items()}

    ds_expected = xr.Dataset(
        {
            "a": xr.DataArray(
                [1, 2], dims=["member_id"], coords={"member_id": ["a", "b"]}
            ).expand_dims("x"),
            "b": xr.DataArray(
                [3, 4], dims=["member_id"], coords={"member_id": ["a", "b"]}
            ).expand_dims("x"),
        },
        attrs={"source_id": "a", "experiment_id": "a"},
    )

    _, ds_combined = merge_variables(concat_members(ddict)).popitem()
    xr.testing.assert_allclose(ds_expected.squeeze(), ds_combined.squeeze())

    with pytest.warns(UserWarning) as warninfo:
        _, ds_combined = concat_members(merge_variables(ddict)).popitem()
    xr.testing.assert_allclose(ds_expected.squeeze(), ds_combined.squeeze())

    msg = (
        "Match attributes ['grid_label', 'table_id'] not found in any of the datasets."
    )
    assert msg in warninfo[0].message.args[0]
