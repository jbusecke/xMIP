from .preprocessing import rename_cmip6
import warnings

def parse_static_thkcello(ds, debug=False):
    ds = ds.copy()
    if debug:
        print(ds.attrs['source_id'])
    dz = ds['lev_bounds'].diff('bnds').squeeze()
    if 'bnds' in dz.dims:
        dz = dz.drop('bnds')
    
    # make sure the new array is 'clean'
    dz = dz.reset_coords(drop=True)
    dz.attrs = {}
    if debug:
        if len(dz.dims)>1:
            print(dz)
    ds.coords['thkcello'] = dz
    
    return ds

def extract_all_availabe_metrics(col, grid_label, source_id, varname='areacello', **kwargs):
    #Find an area for as many models as possible (no matter in which scenario, ensemble etc)
    subset = col.search(variable_id=[varname], grid_label='gn', source_id=source_id)

    # remove dupes from the dataframe
    df = subset.df.drop_duplicates(subset='source_id')
    subset.df = df
    
#     return subset.to_dataset_dict(preprocess=preprocess)# activate this when intake_esm is working
    return subset.to_dataset_dict(**kwargs)

def extract_static_metric(col, grid_label, source_id, varname='areacello', preprocess=None, verbose=False, squeeze=True):
    #Find an area for as many models as possible (no matter in which scenario, ensemble etc)
    subset = col.search(variable_id=[varname], grid_label=grid_label, source_id=source_id)
    
    if verbose:
        print(len(subset.df))
        print(subset)
        
    #
    if len(subset.df) < 1:
        warnings.warn('No area detected for source id: `%s`' %source_id)
        return None
    else:
        # remove dupes from the dataframe
        df = subset.df.drop_duplicates(subset='source_id')
        if verbose:
            print(df)
        subset.df = df
        if verbose:
            print('preprocessing')
            
        metric_dict = subset.to_dataset_dict(preprocess=preprocess)

        # some metrics have other dimensions like `member_id`, squeeze these out
        if squeeze:
            metric_dict = {k:ds.squeeze(drop=True) for k, ds in metric_dict.items()}

        if len(metric_dict) == 0:
            warnings.warn('No metric [%s] found for source_id [%s] and grid_label[%s]' %(varname, source_id, grid_label))
            return None
        elif len(metric_dict) > 1:
            print(metric_dict)
            raise RuntimeError('Something went reaalllly wrong. Metric dict should only have one key')
        else:
            return metric_dict[list(metric_dict.keys())[0]]
    
def parse_metrics(data_dict, col, varname='areacello', preprocess=None, rename=False):
    """parse matching static metrics for each element of `data_dict`. 
    preprocessing shouls eventually eliminate renam...once the """
    data_dict_parsed = {}
    for k, ds in data_dict.items():
        metric = extract_static_metric(col, ds.attrs['grid_label'], ds.attrs['source_id'], preprocess=preprocess)
        if not metric is None:
            if rename:
                metric = rename_cmip6(metric)
            # strip all coords and attributes from metric
            metric = metric.squeeze().reset_coords(drop=True)
            ds.coords[varname] = metric[varname]
            data_dict_parsed[k] = ds
    return data_dict_parsed