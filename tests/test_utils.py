import pytest

from cmip6_preprocessing.utils import google_cmip_col


pytest.importorskip("intake")


def test_google_cmip_col():
    col = google_cmip_col()
    assert col.catalog_file == "https://storage.googleapis.com/cmip6/pangeo-cmip6.csv"
