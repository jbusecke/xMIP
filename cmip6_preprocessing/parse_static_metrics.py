from .preprocessing import rename_cmip6
import warnings

def extract_all_availabe_metrics(col, grid_label, source_id, varname='areacello', **kwargs):
    #Find an area for as many models as possible (no matter in which scenario, ensemble etc)
    subset = col.search(variable_id=[varname], grid_label='gn', source_id=source_id)

    # remove dupes from the dataframe
    df = subset.df.drop_duplicates(subset='source_id')
    subset.df = df
    
#     return subset.to_dataset_dict(preprocess=preprocess)# activate this when intake_esm is working
    return subset.to_dataset_dict(**kwargs)

def extract_static_metric(col, grid_label, source_id, varname='areacello', preprocess=None):
    #Find an area for as many models as possible (no matter in which scenario, ensemble etc)
    subset = col.search(variable_id=[varname], grid_label='gn', source_id=source_id)

    # remove dupes from the dataframe
    df = subset.df.drop_duplicates(subset='source_id')
    subset.df = df
    
#     metric_dict = subset.to_dataset_dict(preprocess=preprocess) # reactivate once intake_esm is ready
    metric_dict = subset.to_dataset_dict()
    
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
        metric = extract_static_metric(col, ds.attrs['grid_label'], ds.attrs['source_id'])
        if not metric is None:
            if rename:
                metric = rename_cmip6(metric)
            # strip all coords and attributes from metric
            metric = metric.squeeze().reset_coords(drop=True)
            ds.coords[varname] = metric[varname]
            data_dict_parsed[k] = ds
    return data_dict_parsed