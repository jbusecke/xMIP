===============================
cmip6_preprocessing
===============================


.. image:: https://img.shields.io/travis/jbusecke/cmip6_preprocessing.svg
        :target: https://travis-ci.org/jbusecke/cmip6_preprocessing
.. image:: https://circleci.com/gh/jbusecke/cmip6_preprocessing.svg?style=svg
    :target: https://circleci.com/gh/jbusecke/cmip6_preprocessing
.. image:: https://codecov.io/gh/jbusecke/cmip6_preprocessing/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/jbusecke/cmip6_preprocessing


Frustrated with how 'dirty' CMIP6 data still is? Do you just want to run a simple (or complicated) analysis on various models and end up having to write logic for each seperate case? Then this package is for you.

Developed during the [cmip6-hackathon](https://cmip6hack.github.io/#/) this package provides utility functions that play nicely with [intake-esm](https://github.com/NCAR/intake-esm).

We currently support the following functions

1. Fix inconsistent naming of dimensions and coordinates
2. Fix inconsistent values,shape and dataset location of coordinates
3. Homogenize longitude conventions
4. Fix inconsistent units

The following issues are under development:
1. Reconstruct/find grid metrics
2. Arrange different variables on their respective staggered grid, so they can work seamlessly with [xgcm](https://xgcm.readthedocs.io/en/latest/)

Please check out the [tutorial](notebooks/tutorial.ipynb) for some examples using the [pangeo cloud](ocean.pangeo.io).
