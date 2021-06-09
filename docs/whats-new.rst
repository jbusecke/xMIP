.. currentmodule:: cmip6_preprocessing

What's New
===========
.. _whats-new.0.2.1:


v0.3.0 (2021/6/9)
-------------------

New Features
~~~~~~~~~~~~
- Added `postprocessing` module and added ability to parse metrics to multiple datasets in a dictionary (:pull:`110`, :pull:`117`). By `Julius Busecke <https://github.com/jbusecke>`_


Internal Changes
~~~~~~~~~~~~~~~~

- Refactored CI internals, added dependabot, and some updated failcases (:pull:`121`, :pull:`128`, :pull:`129`, :pull:`133`, :pull:`134`, :pull:`135`). By `Julius Busecke <https://github.com/jbusecke>`_

.. _whats-new.0.2.0:

v0.2.0 (2021/4/9)
-----------------

Breaking changes
~~~~~~~~~~~~~~~~
- Removed `replace_x_y_nominal_lat_lon` from `combined_preprocessing` due to ongoing performance issues with dask (:issue:`75`, :issue:`85`, :issue:`94`) (:pull:`104`). By `Julius Busecke <https://github.com/jbusecke>`_
- Further refactor of `replace_x_y_nominal_lat_lon`, which avoids missing values in the dimension coordinates (:issue:`66`) (:pull:`79`). By `Julius Busecke <https://github.com/jbusecke>`_

- Consistent treatment of cf-style bounds. The combination of `parse_lon_lat_bounds`,`maybe_convert_bounds_to_vertex`, `maybe_convert_vertex_to_bounds`, and `sort_vertex_order` applied on the dataset, assures that all datasets have both conventions available and the vertex order is the same. By `Julius Busecke <https://github.com/jbusecke>`_

- New implementation of `replace_x_y_nominal_lat_lon`, which avoids duplicate values in the derived dimensions (:issue:`34`) (:pull:`35`). By `Julius Busecke <https://github.com/jbusecke>`_

New Features
~~~~~~~~~~~~
- Create merged region masks with :py:func:`merged_mask` (:pull:`18`). By `Julius Busecke <https://github.com/jbusecke>`_


Bug fixes
~~~~~~~~~
- Updated cmip6 catalog location for the pangeo gc archive (:issue:`80`) (:pull:`81`). By `Julius Busecke <https://github.com/jbusecke>`_


Documentation
~~~~~~~~~~~~~
- Sphinx/RTD documentation, including contributor guide and new logo 🤗. (:issue:`27`) (:pull:`99`).

Internal Changes
~~~~~~~~~~~~~~~~
- Adds options to skip extensive cloud ci by using [skip-ci] in commit message. Adds the ability to cancel previous GHA jobs to prevent long wait times for rapid pushes. (:pull:`99`) By `Julius Busecke <https://github.com/jbusecke>`_.

-  Add `ni` and `nj` to the `rename_dict` dictionary in _preprocessing.py_ as dimensions to be corrected (:pull:`54`). By `Markus Ritschel <https://github.com/markusritschel>`_


.. _whats-new.0.1.2:

v0.1.2
------


New Features
~~~~~~~~~~~~
- Added more models, now supporting both ocean and atmospheric output for :py:func:`combined_preprocessing` (:pull:`14`). By `Julius Busecke <https://github.com/jbusecke>`_



.. _whats-new.0.1.0:

v0.1.0 (2/21/2020)
----------------------

Initial release.
