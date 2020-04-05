# This module tests data directly from the pangeo google cloud storage
import pytest
import numpy as np
from cmip6_preprocessing.preprocessing import combined_preprocessing


@pytest.fixture
def col():
    import intake

    return intake.open_esm_datastore(
        "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"
    )


def test_replace_x_y_nominal_lat_lon(col):
    cat = col.search(
        source_id="MPI-ESM1-2-HR",
        experiment_id="historical",
        variable_id="o2",
        table_id="Omon",
        member_id="r1i1p1f1",
    )
    print(cat.df)
    ddict = cat.to_dataset_dict(
        zarr_kwargs={"consolidated": True}, preprocess=combined_preprocessing
    )
    ds = ddict["CMIP.MPI-M.MPI-ESM1-2-HR.historical.Omon.gn"]
    # check all dims for duplicates
    for di in ds.dims:
        print(di)
        assert len(ds[di]) == len(np.unique(ds[di]))
