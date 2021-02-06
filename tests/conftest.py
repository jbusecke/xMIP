import pytest


def pytest_addoption(parser):
    parser.addoption("--cat", action="store", default="main")
    parser.addoption("--vi", action="store", default="thetao")
    parser.addoption("--gl", action="store", default="gn")
    parser.addoption("--ei", action="store", default="historical")
    parser.addoption("--models", action="store", default="all")
