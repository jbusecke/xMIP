import pytest

from cmip6_preprocessing.utils import google_cmip_col, model_id_match


pytest.importorskip("intake")


def test_google_cmip_col():
    col = google_cmip_col()
    assert col.catalog_file == "https://storage.googleapis.com/cmip6/pangeo-cmip6.csv"


def test_model_id_match():

    # wrong amount of elements
    with pytest.raises(ValueError):

        model_id_match([("A", "a", "aa"), ("A", "a", "aa", "aaa")], ("A", "a", "aa"))

    with pytest.raises(ValueError):
        model_id_match([("A", "a", "aa"), ("B", "a", "aa")], ("A", "a", "aa", "aaa"))

    assert model_id_match([("A", "a", "aa"), ("B", "a", "aa")], ("A", "a", "aa"))
    assert ~model_id_match([("A", "a", "aa"), ("B", "a", "aa")], ("C", "a", "aa"))
    assert ~model_id_match([("A", "a", "aa"), ("B", "a", "aa")], ("AA", "a", "aa"))
    assert ~model_id_match([("AA", "a", "aa"), ("B", "a", "aa")], ("A", "a", "aa"))
    assert model_id_match([("*", "a", "aa")], ("whatever", "a", "aa"))
    assert model_id_match([(["bb", "b"], "a", "aa")], ("b", "a", "aa"))
    assert model_id_match(
        [(["bb", "b"], "a", "aa"), (["bb", "b"], "c", "cc")], ("bb", "a", "aa")
    )
