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


def all_models():
    col = intake.open_esm_datastore(
        "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"
    )
    df = col.df
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


@pytest.mark.parametrize("grid_label", ["gr", "gn"])
@pytest.mark.parametrize("experiment_id", ["historical"])
@pytest.mark.parametrize("variable_id", ["o2", "thetao"])
@pytest.mark.parametrize("source_id", all_models())
def test_lat_lon(col, source_id, experiment_id, grid_label, variable_id):
    cat = col.search(
        source_id=source_id,
        experiment_id=experiment_id,
        variable_id=variable_id,
        table_id="Omon",
        grid_label=grid_label,
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
            assert ds.lon.min() >= 0
            assert ds.lon.max() <= 360
            assert ds.lat.min() >= -90
            assert ds.lat.max() <= 90

    else:
        pytest.xfail("Model data not available")
