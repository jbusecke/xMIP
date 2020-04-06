# This module tests data directly from the pangeo google cloud storage
import pytest
import numpy as np
import intake
from cmip6_preprocessing.preprocessing import combined_preprocessing

pytest.importorskip("gcsfs")


@pytest.fixture
def col():
    return intake.open_esm_datastore(
        "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"
    )


# define a class with variable fixture...


def all_models(var, grid_label):
    col = intake.open_esm_datastore(
        "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"
    )
    # this is clunky...
    df = col.df[[co in [var] for co in col.df["variable_id"]]]
    df = df[[co in [grid_label] for co in df["grid_label"]]]
    all_models = df["source_id"].unique()

    # TODO: finally get IPSL model to run and release this
    return [m for m in all_models if "IPSL" not in m]


def _diagnose_doubles(data):
    """displays non-unique entries in data"""
    u, idx = np.unique(data, return_index=True)
    missing = np.array([i for i in np.arange(len(data)) if i not in idx])
    if len(missing) > 0:
        missing_values = data[missing]
        print(f"Missing values Indicies[{missing}]/ Values[{missing_values}]")


@pytest.mark.parametrize("experiment_id", ["historical", "piControl", "ssp585"])
@pytest.mark.parametrize("source_id", all_models("o2", "gn"))
def test_gn_replace_x_y_nominal_lat_lon(col, source_id, experiment_id):
    cat = col.search(
        source_id=source_id,
        experiment_id=experiment_id,
        variable_id="o2",
        table_id="Omon",
    )
    ddict = cat.to_dataset_dict(
        zarr_kwargs={"consolidated": True, "decode_times": False},
        preprocess=combined_preprocessing,
    )
    if len(ddict) > 0:
        _, ds = ddict.popitem()
        # check all dims for duplicates
        for di in ds.dims:
            print(di)
            _diagnose_doubles(ds[di].load().data)
            assert len(ds[di]) == len(np.unique(ds[di]))

    else:
        pytest.xfail("Model data not available")


@pytest.mark.parametrize("experiment_id", ["historical", "piControl", "ssp585"])
@pytest.mark.parametrize("source_id", all_models("o2", "gr"))
def test_gr_replace_x_y_nominal_lat_lon(col, source_id, experiment_id):
    cat = col.search(
        source_id=source_id,
        experiment_id=experiment_id,
        variable_id="o2",
        table_id="Omon",
    )
    ddict = cat.to_dataset_dict(
        zarr_kwargs={"consolidated": True, "decode_times": False},
        preprocess=combined_preprocessing,
    )
    if len(ddict) > 0:
        _, ds = ddict.popitem()
        # check all dims for duplicates
        for di in ds.dims:
            print(di)
            _diagnose_doubles(ds[di].load().data)
            assert len(ds[di]) == len(np.unique(ds[di]))

    else:
        pytest.xfail("Model data not available")
