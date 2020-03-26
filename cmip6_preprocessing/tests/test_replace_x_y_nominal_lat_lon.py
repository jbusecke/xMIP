import pytest
import intake
import numpy as np
import xarray as xr
from cmip6_preprocessing.preprocessing import replace_x_y_nominal_lat_lon


def test_replace_x_y_nominal_lat_lon():
    x = np.linspace(0,360,720)
    y = np.linspace(-90, 90, 360)
    data = np.random.rand(len(x), len(y))
    ds = xr.DataArray(data, coords=[('x', x), ('y', y)]).to_dataset(name='data')
    ds['lon'] = ds['x'] * xr.ones_like(ds['y'])
    ds['lat'] = xr.ones_like(ds['x']) * ds['y']
    replaced_ds = replace_x_y_nominal_lat_lon(ds)
    