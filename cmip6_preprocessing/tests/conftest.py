import pytest


def pytest_addoption(parser):
    parser.addoption("--variable_id_global", action="store", default="thetao")
    parser.addoption("--grid_label_global", action="store", default="gn")
    parser.addoption("--experiment_id_global", action="store", default="historical")