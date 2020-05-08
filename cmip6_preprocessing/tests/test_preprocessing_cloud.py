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


def all_models():
    col = intake.open_esm_datastore(
        "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"
    )
    df = col.df
    all_models = df["source_id"].unique()

    # TODO: finally get IPSL model to run and release this
    # TODO: Allow the AWI regridded model output for the preprocessing module
    return [m for m in all_models if (("IPSL" not in m) & ("AWI" not in m))]
    # return [m for m in all_models if "MIROC" in m]


def _diagnose_doubles(data):
    """displays non-unique entries in data"""
    u, idx = np.unique(data, return_index=True)
    missing = np.array([i for i in np.arange(len(data)) if i not in idx])
    if len(missing) > 0:
        missing_values = data[missing]
        print(f"Missing values Indicies[{missing}]/ Values[{missing_values}]")


# These are too many tests. Perhaps I could load all the data first and then
# test each dict item?


@pytest.mark.parametrize("grid_label", ["gr", "gn"])
@pytest.mark.parametrize("experiment_id", ["historical"])
@pytest.mark.parametrize("variable_id", ["o2", "thetao"])
@pytest.mark.parametrize("source_id", all_models())
def test_preprocessing_combined(col, source_id, experiment_id, grid_label, variable_id):
    cat = col.search(
        source_id=source_id,
        experiment_id=experiment_id,
        variable_id=variable_id,
        # member_id="r1i1p1f1",
        table_id="Omon",
        grid_label=grid_label,
    )

    ddict_raw = cat.to_dataset_dict(
        zarr_kwargs={"consolidated": True, "decode_times": False},
        preprocess=None,
        storage_options={"token": "anon"},
    )
    if len(ddict_raw) > 0:
        _, ds_raw = ddict_raw.popitem()
        print(ds_raw)

    ddict = cat.to_dataset_dict(
        zarr_kwargs={"consolidated": True, "decode_times": False},
        preprocess=combined_preprocessing,
        storage_options={"token": "anon"},
    )
    if len(ddict) > 0:

        _, ds = ddict.popitem()

        ##### Check for dim duplicates
        # check all dims for duplicates
        # for di in ds.dims:
        # for now only test a subset of the dims. TODO: Add the bounds once they
        # are cleaned up.
        for di in ["x", "y", "lev", "time"]:
            print(di)
            if di in ds.dims:
                _diagnose_doubles(ds[di].load().data)
                assert len(ds[di]) == len(np.unique(ds[di]))
        assert ds.lon.min().load() >= 0
        assert ds.lon.max().load() <= 360
        assert ds.lat.min().load() >= -90
        assert ds.lat.max().load() <= 90
        # make sure lon and lat are 2d
        assert len(ds.lon.shape) == 2
        assert len(ds.lat.shape) == 2
        if "vertex" in ds.dims:
            np.testing.assert_allclose(ds.vertex.data, np.arange(4))

        print(ds)
        print(ds.lon_bounds.load())

        if source_id == "FGOALS-f3-L":
            pytest.skip("`FGOALS-f3-L` does not come with lon/lat bounds")
        else:
            ####Check for existing bounds and verticies
            for co in ["lon_bounds", "lat_bounds", "lon_verticies", "lat_verticies"]:
                assert co in ds.coords
                assert set(["x", "y"]).issubset(set(ds[co].dims))

            #### Check the order of the vertex
            test_vertex = ds.isel(x=len(ds.x) // 2, y=len(ds.y) // 2)
            print(test_vertex.lon_verticies.load())

            assert test_vertex.lon_verticies.isel(
                vertex=0
            ) < test_vertex.lon_verticies.isel(vertex=3)
            assert test_vertex.lon_verticies.isel(
                vertex=1
            ) < test_vertex.lon_verticies.isel(vertex=2)

            assert test_vertex.lat_verticies.isel(
                vertex=0
            ) < test_vertex.lat_verticies.isel(vertex=1)
            assert test_vertex.lat_verticies.isel(
                vertex=3
            ) < test_vertex.lat_verticies.isel(vertex=2)

            assert np.all(ds.lon_bounds.diff("bnds") > 0)
            assert np.all(ds.lat_bounds.diff("bnds") > 0)

    else:
        pytest.xfail("Model data not available")
