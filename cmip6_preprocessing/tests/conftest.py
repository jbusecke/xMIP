import pytest


def pytest_addoption(parser):
    parser.addoption("--vi", action="store", default="thetao")
    parser.addoption("--gl", action="store", default="gn")
    parser.addoption("--ei", action="store", default="historical")
