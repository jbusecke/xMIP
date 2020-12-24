import pytest


def pytest_addoption(parser):
    parser.addoption("--variable_id", action="store", default="thetao")
    parser.addoption("--grid_label", action="store", default="gn")
    parser.addoption("--experiment_id", action="store", default="historical")
    # parser.addoption("--experiement", action="store", default="historical")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".

    for name in ["variable_id", "grid_label", "experiment_id"]:  # ,

        option_value = getattr(metafunc.config.option, name)

        if isinstance(option_value, str):
            option_value = [option_value]

        if name in metafunc.fixturenames and option_value is not None:
            metafunc.parametrize(name, option_value)
