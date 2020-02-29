import pytest
import intake
import pandas as pd
from cmip6_preprocessing.preprocessing import cmip6_renaming_dict

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

        # check that there are no extra entries
        assert len(set(cmip6_renaming_dict()[model].keys()) - set(required_coords + additonal_coords)) == 0
    
