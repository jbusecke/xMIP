import warnings
import xarray as xr
import numpy as np

def _decode_wrapper(da):
    return xr.decode_cf(da.to_dataset(name='time'), use_cftime=True).time

def branched_time_adjust(source, target, force=False):
    """Adjusts the source time convention to match the target using the meta data of target.
    This is helpful to align e.g. piControl runs with historic or scenario timelines. 
    
    As an example: The GFDL-CM4 historic run, was branched of the piControl in year 151, but the timestamp of the historic run starts at 1850. 
    branched(ds_control, ds_historical) will adjust the time of the control run, so that year 151 is now 1850.

    Parameters
    ----------
    source : xarray.Dataset
        The source dataset. This is the parent of the target dataset
    target : xarray.Dataset
        The target dataset.
    force : bool, optional
        If enabled (True) the data is not checked for parent chile consistency (default: False)
        
    Returns
    -------
    source_corrected : xr.Dataset
        the source dataset with adjusted time dimension to fit the target convention.

    """
    source = source.copy()
    target = target.copy()
    
    # retrieve calendars
    source_calendar = source.time.data[0].calendar
    target_calendar = target.time.data[0].calendar
    
    if source_calendar != target_calendar:
        warnings.warn(f"Source and Target did not have the same calendar ({source_calendar},{target_calendar}) [{source.attrs['source_id']}]. Returning unmodified datasets")
        return source
    else:
        calendar = source_calendar
        # make sure the source and target are compatible
        source_id = source.attrs['experiment_id'].replace(' ', '')
        target_parent_id = target.attrs['parent_experiment_id'].replace(' ', '')
        # this is necessary because the `CNRM-CM6-1-HR` model has values like this: `p i C o n t r o l`
        
        if not source_id == target_parent_id:
            raise ValueError(f"Source({source.attrs['source_id']}) not consistent with Target({target.attrs['source_id']}) [source experiment_id: {source_id} | target parent_experiment_id: {target_parent_id}]. ")

        # this is the timestamp of the branch in the conventions of source
        branch_time_source = xr.DataArray(target.attrs['branch_time_in_parent'])
#         branch_time_source.attrs = target.time.attrs
        branch_time_source.attrs['calendar'] = calendar
        branch_time_source.attrs['units'] = target.attrs['parent_time_units']
        branch_time_source = _decode_wrapper(branch_time_source)
        
        # now we assume that the historical run is present right from the branching point. E.g. time[0] is the branching point
        branch_time_target = target.time[0]
        
        # # now remove the time since year 0 to the branch point from the source time (setting branch date to year 0)
        delta_time = branch_time_target.data - branch_time_source.data
        source.coords['time'].data = source.time.data + delta_time
        source = xr.decode_cf(source)
        return source
        
#         # now remove the time since year 0 to the branch point from the source time (setting branch date to year 0)
        
#         null_time = _decode_wrapper(xr.DataArray(0, attrs={'units':target.attrs['parent_time_units'], 'calendar':calendar}))
#         delta_time_origin = branch_time_source.data - null_time.data
        
        
#         # Finally add the branch time of the target and set as new time
#         source.coords['time'].data = source.time.data - (source.time[0].data + delta_time_origin) + branch_time_target.data
#         source = xr.decode_cf(source)
#         return source

# build dependency graph for branched runs (do this later, at the moment I do have all experiments)
# TODO: add

# Wrapper function to adjust all datasets in dictionary 
#(recursively, but that wont be necessary at first, add that with the dependency detection)
def unify_branched_time(in_dict, experiment_id_order, verbose=False):
    """Wrapper function to convert all runs in an `intake-esm` generated data dictionary `in_dict` to a commmon time convention.

    Parameters
    ----------
    in_dict : dict
        Dictionary of xarray.Datasets, created by intake-esm `.to_dataset_dict()`
    experiment_id_order : list
        List of `experiment_id` values in descending parent order, e.g. values towards the right have to be parents of the previous values.
    verbose: bool, optional
        Prints versbose output (Default:False).

    Returns
    -------
    out_dict:
        Dictionary of xarray.Datasets with adjusted time conventions.

    """
    out_dict = {}
    source_ids = np.unique([ds.attrs['source_id'] for ds in in_dict.values()])
    for source_id in source_ids:
        if verbose:
            print(source_id)
        for oi in range(len(experiment_id_order)-1):
            target_experiment = experiment_id_order[oi]
            source_experiment = experiment_id_order[oi+1]
            target_keys = [k for k,ds in in_dict.items() if ((ds.attrs['source_id'] == source_id) &
                                                             (ds.attrs['experiment_id'] == target_experiment))]
            source_keys = [k for k,ds in in_dict.items() if ((ds.attrs['source_id'] == source_id) &
                                                             (ds.attrs['experiment_id'] == source_experiment))]
            
            # TODO: I do this repeatedly. I should probably factor this out in a utils module.
            if (len(target_keys) != 1) | (len(source_keys) != 1):
                raise RuntimeError(f"Found more than 1 key or no key for source:{source_keys} or target:{target_keys}. This would need some additional logic")
            else:
                target_key = target_keys[0]
                source_key = source_keys[0]
            if verbose:
                print(f"target:{target_key}")
                print(f"source:{source_key}")
            out_dict[source_key] = branched_time_adjust(in_dict[source_key],in_dict[target_key])
    return out_dict