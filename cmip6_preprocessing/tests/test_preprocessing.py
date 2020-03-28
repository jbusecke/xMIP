import pytest
import intake
import pandas as pd
import numpy as np
import xarray as xr
from cmip6_preprocessing.preprocessing import cmip6_renaming_dict, replace_x_y_nominal_lat_lon

# get all available ocean models from the cloud. 
url = 'https://storage.googleapis.com/cmip6/pangeo-cmip6.csv'
df = pd.read_csv(url)
df_ocean =df[(df.table_id=='Omon') + (df.table_id=='Oyr')]
ocean_models = df_ocean.source_id.unique()

# TODO: Need to adapt atmos only models
all_models = ocean_models

required_coords = ['x', 'y', 'lon', 'lat']
additonal_coords = ['lev', "lev_bounds",'bnds', "time_bounds", 'vertex', "lat_bounds", "lon_bounds"]

def test_renaming_dict_keys():
    # check that each model has an entry
    rename_dict = cmip6_renaming_dict()
    assert set(list(rename_dict.keys())) == set(list(all_models))

@pytest.mark.parametrize("model", all_models)
def test_renaming_dict_entry_keys(model):
    if 'AWI' not in model: # excluding the unstructured awi model
        # check that all required coords are there
        for co in  required_coords:
            assert co in cmip6_renaming_dict()[model].keys()
        # check that (if present) x and y contain lon and lat entries
        if 'x' in cmip6_renaming_dict()[model].keys():
            assert 'lon' in cmip6_renaming_dict()[model]['x']
        if 'y' in cmip6_renaming_dict()[model].keys():
            assert 'lat' in cmip6_renaming_dict()[model]['y']

        # check that there are no extra entries
        assert len(set(cmip6_renaming_dict()[model].keys()) - set(required_coords + additonal_coords)) == 0


def test_replace_x_y_nominal_lat_lon():
    x = np.linspace(0,360,720)
    y = np.linspace(-90, 90, 360)
    data = np.random.rand(len(x), len(y))
    ds = xr.DataArray(data, coords=[('x', x), ('y', y)]).to_dataset(name='data')
    ds['lon'] = ds['x'] * xr.ones_like(ds['y'])
    ds['lat'] = xr.ones_like(ds['x']) * ds['y']
    replaced_ds = replace_x_y_nominal_lat_lon(ds)
