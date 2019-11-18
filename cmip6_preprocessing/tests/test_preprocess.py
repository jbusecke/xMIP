import pytest
import intake
import pandas
from cmip6_preprocessing.preprocessing import cmip6_renaming_dict

all_models = set([ 'BCC-CSM2-MR', 'BCC-ESM1', 'CAMS-CSM1-0',
       'FGOALS-f3-L', 'CanESM5', 'CNRM-CM6-1', 'CNRM-ESM2-1',
       'E3SM-1-0', 'EC-Earth3-LR', 'EC-Earth3-Veg', 'EC-Earth3',
        'IPSL-CM6A-LR', 'MIROC-ES2L', 'MIROC6',
        'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 'UKESM1-0-LL',
       'GISS-E2-1-G-CC', 'GISS-E2-1-G', 'GISS-E2-1-H', 'CESM2-WACCM',
       'CESM2', 'NorCPM1', 'GFDL-CM4', 'GFDL-ESM4', 'NESM3',
       'SAM0-UNICON', 'MCM-UA-1-0', 'MPI-ESM1-2-HR','NorESM1-F','FGOALS-g3','MRI-ESM2-0'])

# omit this one
#'AWI-CM-1-1-MR',

required_coords = ['x', 'y', 'lon', 'lat', 'lev', "lev_bounds", "time_bounds"]
additonal_coords = ['bnds', 'vertex', "lat_bounds", "lon_bounds"]

def test_renaming_dict_keys():
    # check that each model has an entry
    rename_dict = cmip6_renaming_dict()
    # omit this weird one
    del rename_dict['AWI-CM-1-1-MR']
    assert set(list(rename_dict.keys())) == all_models

@pytest.mark.parametrize("model", all_models)
def test_renaming_dict_entry_keys(model):
    # check that all required coords are there
    for co in  required_coords:
        assert co in cmip6_renaming_dict()[model].keys()
    
    # check that there are no extra entries
    assert len(set(cmip6_renaming_dict()[model].keys()) - set(required_coords + additonal_coords)) == 0
    