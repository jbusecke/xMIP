.. cmip6_preprocessing documentation master file, created by
   sphinx-quickstart on Thu Feb 25 16:11:36 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: images/logo.png

Analysis ready CMIP6 data with Pangeo
=====================================

Modern climate science like the IPCC rely heavily on model inter comparison projects (MIPs). These projects essentially pool together model results from various climate modeling centers around the world, that were run according to specific protocols, in order to compare, for instance, the response of the coupled climate system to changes in forcing.

The vast amount of work that has been put into the standardization of these experiments enables climate scientists to use a wealth of data to answer their specific questions, thus refining future models and increasing our understanding of the complex system that is our home planet.

However, from the viewpoint of analyzing these data, the output is still quite 'dirty' making the quintessential workflow of:

1. Develop a metric/analysis to apply to one model.
2. Run that analysis across all the models  and interpret results.

inherently difficult.

Most of the problems arise from differences in the convention the model output is provided in. This includes, but is not limited to different naming conventions for coordinate variables,  units, grid variables.
`cmip6_preprocessing` aims to provide lightweight tools, that let you get right to the science, without spending hours on cleaning up the data.



Installation
------------

Installation from Conda Forge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install xgcm along with its dependencies is via conda
forge::

    conda install -c conda-forge cmip6_preprocessing

Installation from Pip
^^^^^^^^^^^^^^^^^^^^^

An alternative is to use pip::

    pip install cmip6_preprocessing

Installation from GitHub
^^^^^^^^^^^^^^^^^^^^^^^^

You can get the newest version by installing directly from GitHub::

    pip install git+https://github.com/jbusecke/cmip6_preprocessing.git


Getting Started
---------------

The most basic functionality is provided by the `combined_preprocessing` function. Check out the `tutorial <tutorial.ipynb>`_ for a brief introduction of the basic functionality.


Suggested Workflow
------------------

We aim to provide a flexible solution for many scientific workflows which might need combination of datasets at different 'levels'.

.. image:: images/workflow_diagram.png

The `preprocessing` module deals with 'cleaning' single variable datasets (e.g. from a single zarr store in the `pangeo CMIP6 cloud data <https://pangeo-data.github.io/pangeo-cmip6-cloud/>`_ or a dataset loaded from mulitple netcdf files on a local server/HPC).

Depending on your science goal, you might need to combine several datasets into members (multi variable datasets) or even further. These combination tasks are facilitated by the `postprocessing` module. This provides the ability to 'match and combine' datasets based on their attributes. For more detailed examples please check out the `Postprocessing` section.


.. I need to check out how to link the API sections and from within notebooks properly. Look into https://myst-nb.readthedocs.io/en/latest/


Contents
--------

.. toctree::
   :maxdepth: 1

   tutorial
   postprocessing
   regionmask
   contributor-guide
   api
   whats-new

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
