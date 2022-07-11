# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import pathlib
import sys


print("python exec:", sys.executable)
print("sys.path:", sys.path)
root = pathlib.Path(__file__).parent.parent.absolute()
os.environ["PYTHONPATH"] = str(root)
sys.path.insert(0, str(root))

import xmip  # isort:skip
from importlib.metadata import version  # isort:skip

release = version("xmip")
# for example take major/minor/patch
version = ".".join(release.split(".")[:3])

# From https://github.com/pypa/setuptools_scm/#usage-from-sphinx

# -- Project information -----------------------------------------------------

project = "xmip"
copyright = "2021, xmip maintainers"
author = "xmip maintainers"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "recommonmark",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "nbsphinx",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxcontrib.srclinks",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

# link to github issues
extlinks = {
    "issue": ("https://github.com/jbusecke/xmip/issues/%s", "GH#"),
    "pull": ("https://github.com/jbusecke/xmip/issues/%s", "GH#"),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pangeo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
