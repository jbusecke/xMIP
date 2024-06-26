.. currentmodule:: xmip

What's New
===========
.. _whats-new.0.8.0:

v0.8.0 (unreleased)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Add `longitude_bnds` and `latitude_bnds` to `cmip_renaming_dict` (#300). By `Joran Angevaare <https://github.com/JoranAngevaare>`
- Updated pre-commit linting to use ruff (#359). By `Julius Busecke <https://github.com/jbusecke>`
- Modernized packaging workflow, that runs on each PR (#361). By `Julius Busecke <https://github.com/jbusecke>`
- Added 'nvertices' -> 'vertex' to renaming preprocessor (#357). By `Julius Busecke <https://github.com/jbusecke>`
- Updated mamba CI + testing py311/py312 (#360, #362). By `Julius Busecke <https://github.com/jbusecke>`

Bugfixes
~~~~~~~~
- Fixed cyclic interpolation in `_interp_nominal_lon` (#295, #296). By `Joran Angevaare <https://github.com/JoranAngevaare>`

.. _whats-new.0.7.2:

v0.7.3 (unreleased)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Added PR template (#304). By `Julius Busecke <https://github.com/jbusecke>`
- Add `longitude_bnds` and `latitude_bnds` to `cmip_renaming_dict` (#300). By `Joran Angevaare <https://github.com/JoranAngevaare>`

.. _whats-new.0.7.0:

v0.7.0 (2023/01/03)
-------------------

New Features
~~~~~~~~~~~~
- :py:func:`~xmip.postprocessing.match_metrics` Now allows more flexible metric matching (accepting e.g. already merged members) + better error for missing match_attrs (#275). By `Julius Busecke <https://github.com/jbusecke>`_
- Postprocessing functions can now easily be nested on top of each other (#187). By `Julius Busecke <https://github.com/jbusecke>`_


Breaking Changes
~~~~~~~~~~~~~~~~
- Requires xarray>=0.17.0 and drops support for python 3.6 (#170, #173). By `Julius Busecke <https://github.com/jbusecke>`_
- :py:func:`~xmip.utils.cmip6_dataset_id` not includes the attribute `variable_id` (#166) By `Julius Busecke <https://github.com/jbusecke>`_
- Dropped support for python 3.7 (#268, #267). By `Julius Busecke <https://github.com/jbusecke>`_

Internal Changes
~~~~~~~~~~~~~~~~

- Unit correction logic now uses pint-xarray under the hood (#160, #31).
By `Tom Nicholas <https://github.com/TomNicholas>`_ and `Julius Busecke <https://github.com/jbusecke>`_

- License changed to Apache-2.0 (#272, #256). By `Julius Busecke <https://github.com/jbusecke>`_

Bugfixes
~~~~~~~~
- :py:func:`~xmip.postprocessing.concat_members` now accepts datasets which already have 'member_id' as a dimension (maintain compatibility with recent intake-esm changes) (#277). By `Julius Busecke <https://github.com/jbusecke>`_

- :py:func:`~xmip.postprocessing.match_metrics` now accepts single variables as str input (#229, #245). By `Julius Busecke <https://github.com/jbusecke>`_

- :py:func:`~xmip.postprocessing.concat_members` now returns a dataset with labelled `member_id` dimension (#196 , #197). By `Julius Busecke <https://github.com/jbusecke>`_

- Fixes incompatibility with upstream changes in xarray>=0.19.0 (#173, #174). By `Julius Busecke <https://github.com/jbusecke>`_

- :py:func:`~xmip.drift_removal.match_and_remove_drift` does now work with chunked (dask powered) datasets (#164).By `Julius Busecke <https://github.com/jbusecke>`_

Internal Changes
~~~~~~~~~~~~~~~~

- Unit correction logic now uses pint-xarray under the hood (#160, #31).
By `Tom Nicholas <https://github.com/TomNicholas>`_ and `Julius Busecke <https://github.com/jbusecke>`_


.. _whats-new.0.5.0:

v0.5.0 (2021/7/9)
-------------------

New Features
~~~~~~~~~~~~
- :py:func:`~xmip.postprocessing.interpolate_grid_labels` enables batch combination of different grid_labels
(e.g. from native to regridded and vice versa) using xesmf (#161). By `Julius Busecke <https://github.com/jbusecke>`_

- :py:func:`~xmip.drift_removal.match_and_remove_drift` enables batch detrending/drift-drift_removal
from a dictionary of datasets (#155). By `Julius Busecke <https://github.com/jbusecke>`_

.. _whats-new.0.4.0:

v0.4.0 (2021/6/9)
-------------------

New Features
~~~~~~~~~~~~

- Started implementing metadata fixes in `combined_preprocessing` (#147). By `Julius Busecke <https://github.com/jbusecke>`_

- Added `drift_removal` which adds ability to align time of branched runs and remove drift from the parent (e.g. control) run (#126, #148). By `Julius Busecke <https://github.com/jbusecke>`_

.. _whats-new.0.3.0:

v0.3.0 (2021/6/9)
-------------------

New Features
~~~~~~~~~~~~
- Added `postprocessing` module and added ability to parse metrics to multiple datasets in a dictionary (#110, #117). By `Julius Busecke <https://github.com/jbusecke>`_


Internal Changes
~~~~~~~~~~~~~~~~

- Refactored CI internals, added dependabot, and some updated failcases (#121, #128, #129, #133, #134, #135). By `Julius Busecke <https://github.com/jbusecke>`_

.. _whats-new.0.2.0:

v0.2.0 (2021/4/9)
-----------------

Breaking changes
~~~~~~~~~~~~~~~~
- Removed `replace_x_y_nominal_lat_lon` from `combined_preprocessing` due to ongoing performance issues with dask (#75, #85, #94) (#104). By `Julius Busecke <https://github.com/jbusecke>`_
- Further refactor of `replace_x_y_nominal_lat_lon`, which avoids missing values in the dimension coordinates (#66) (#79). By `Julius Busecke <https://github.com/jbusecke>`_

- Consistent treatment of cf-style bounds. The combination of `parse_lon_lat_bounds`,`maybe_convert_bounds_to_vertex`, `maybe_convert_vertex_to_bounds`, and `sort_vertex_order` applied on the dataset, assures that all datasets have both conventions available and the vertex order is the same. By `Julius Busecke <https://github.com/jbusecke>`_

- New implementation of `replace_x_y_nominal_lat_lon`, which avoids duplicate values in the derived dimensions (#34) (#35). By `Julius Busecke <https://github.com/jbusecke>`_

New Features
~~~~~~~~~~~~
- Create merged region masks with :py:func:`merged_mask` (#18). By `Julius Busecke <https://github.com/jbusecke>`_


Bug fixes
~~~~~~~~~
- Updated cmip6 catalog location for the pangeo gc archive (#80) (#81). By `Julius Busecke <https://github.com/jbusecke>`_


Documentation
~~~~~~~~~~~~~
- Sphinx/RTD documentation, including contributor guide and new logo 🤗. (#27) (#99).

Internal Changes
~~~~~~~~~~~~~~~~
- Adds options to skip extensive cloud ci by using [skip-ci] in commit message. Adds the ability to cancel previous GHA jobs to prevent long wait times for rapid pushes. (#99) By `Julius Busecke <https://github.com/jbusecke>`_.

-  Add `ni` and `nj` to the `rename_dict` dictionary in _preprocessing.py_ as dimensions to be corrected (#54). By `Markus Ritschel <https://github.com/markusritschel>`_


.. _whats-new.0.1.2:

v0.1.2
------


New Features
~~~~~~~~~~~~
- Added more models, now supporting both ocean and atmospheric output for :py:func:`combined_preprocessing` (#14). By `Julius Busecke <https://github.com/jbusecke>`_



.. _whats-new.0.1.0:

v0.1.0 (2/21/2020)
----------------------

Initial release.
