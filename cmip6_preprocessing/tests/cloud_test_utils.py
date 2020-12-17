import pytest
import contextlib
import xarray as xr
import numpy as np
import intake
import fsspec
import itertools
from cmip6_preprocessing.preprocessing import combined_preprocessing
from cmip6_preprocessing.grids import combine_staggered_grid

pytest.importorskip("gcsfs")


def col():
    return intake.open_esm_datastore(
        "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"
    )


def diagnose_doubles(data):
    """displays non-unique entries in data"""
    _, idx = np.unique(data, return_index=True)
    missing = np.array([i for i in np.arange(len(data)) if i not in idx])
    if len(missing) > 0:
        missing_values = data[missing]
        print(f"Missing values Indicies[{missing}]/ Values[{missing_values}]")


def xfail_wrapper(specs, fail_specs):
    # fail out if there is a fail spec that is not in the list
    # unknown_fail_specs = [fail for fail in fail_specs if fail not in specs]
    # if len(unknown_fail_specs) > 0:
    #     raise ValueError(
    #         f"Found fail specs that are not part of the testing {unknown_fail_specs}"
    #     )
    wrapped_specs = []
    for spec in specs:
        if spec in fail_specs:
            wrapped_specs.append(
                pytest.param(*spec, marks=pytest.mark.xfail(strict=True))
            )
        else:
            wrapped_specs.append(spec)
    return wrapped_specs


def data(source_id, variable_id, experiment_id, grid_label, use_intake_esm):
    zarr_kwargs = {
        "consolidated": True,
        "decode_times": False,
        # "decode_times": True,
        # "use_cftime": True,
    }

    cat = col().search(
        source_id=source_id,
        experiment_id=experiment_id,
        variable_id=variable_id,
        # member_id="r1i1p1f1",
        table_id="Omon",
        grid_label=grid_label,
    )

    if len(cat.df["zstore"]) > 0:
        if use_intake_esm:
            ddict = cat.to_dataset_dict(
                zarr_kwargs=zarr_kwargs,
                preprocess=combined_preprocessing,
                storage_options={"token": "anon"},
            )
            _, ds = ddict.popitem()
        else:
            ##### debugging options
            # @charlesbluca suggested this to make this work in GHA
            # https://github.com/jbusecke/cmip6_preprocessing/pull/62#issuecomment-741928365
            mm = fsspec.get_mapper(
                cat.df["zstore"][0]
            )  # think you can pass in storage options here as well?
            ds_raw = xr.open_zarr(mm, **zarr_kwargs)
            print(ds_raw)
            ds = combined_preprocessing(ds_raw)
    else:
        ds = None

    return ds, cat


def all_models():
    df = col().df
    all_models = df["source_id"].unique()
    all_models = tuple(np.sort(all_models))
    return all_models


def full_specs():
    grid_labels = tuple(["gn", "gr"])
    experiment_ids = tuple(["historical", "ssp585"])
    variable_ids = tuple(["thetao", "o2"])

    test_specs = list(
        itertools.product(
            *[
                all_models(),
                variable_ids,
                experiment_ids,
                grid_labels,
            ]
        )
    )
    return test_specs
