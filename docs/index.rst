.. cmip6_preprocessing documentation master file, created by
   sphinx-quickstart on Tue Jul 21 16:08:10 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

cmip6_preprocessing: Making working with model intercomparisons easier since 2019!
==================================================================================

Modern climate science like e.g. the IPCC rely heavily on model inter comparison projects (MIPs). These projects essentially pool together model results from various climate modeling centers around the world, that were run according to specific protocols, in order to compare, for instance, the response of the coupled climate system to changes in forcing.

The vast amount of work that has been put into the standardization of these experiments enables climate scientists to use a wealth of data to answer their specific questions, thus refining future models and increasing our understanding of the complex system that our planet is.

However, from the viewpoint of analyzing these data, the output is still quite 'dirty' making the quintessential workflow of:

1. Develop a metric/analysis to apply to one model
2. Run that analysis across all the models  and interpret results

inherently difficult. 

Most of the problems arise from differences in the convention the model output is provided in. This includes, but is not limited to different naming conventions for coordinate variables,  units, grid variables.



Contents
--------

.. toctree::
   :maxdepth: 1

   tutorial
   regionmask
   whats-new
