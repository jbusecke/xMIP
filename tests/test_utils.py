import pytest

from cmip6_preprocessing.utils import google_cmip_col, model_id_match


pytest.importorskip("intake")


def test_google_cmip_col():
    col = google_cmip_col(catalog="main")
    assert col.catalog_file == "https://storage.googleapis.com/cmip6/pangeo-cmip6.csv"
    col = google_cmip_col(catalog="testing")
    assert (
        col.catalog_file
        == "https://storage.googleapis.com/cmip6/pangeo-cmip6-testing.csv"
    )


def test_model_id_match():

    # wrong amount of elements
    with pytest.raises(ValueError):

        model_id_match([("A", "a", "aa"), ("A", "a", "aa", "aaa")], ("A", "a", "aa"))

    with pytest.raises(ValueError):
        model_id_match([("A", "a", "aa"), ("B", "a", "aa")], ("A", "a", "aa", "aaa"))

    assert model_id_match([("A", "a", "aa"), ("B", "a", "aa")], ("A", "a", "aa"))
    assert ~model_id_match([("A", ["b", "c"], "aa")], ("A", "a", "aa"))
    assert ~model_id_match([("A", ["b", "c"], "aa")], ("A", "a", "aa"))
    assert ~model_id_match(
        [("EC-Earth3-AerChem", ["so"], "historical", "gn")],
        ("EC-Earth3", ["so"], "historical", "gn"),
    )
    assert ~model_id_match([("A", "a", "aa"), ("B", "a", "aa")], ("AA", "a", "aa"))
    assert ~model_id_match([("AA", "a", "aa"), ("B", "a", "aa")], ("A", "a", "aa"))
    assert ~model_id_match([(["AA"], "a", "aa"), ("B", "a", "aa")], ("A", "a", "aa"))
    assert ~model_id_match([(["AA"], ["a"], "aa"), ("B", "a", "aa")], ("A", "a", "aa"))
    assert model_id_match([("*", "a", "aa")], ("whatever", "a", "aa"))
    assert model_id_match([(["bb", "b"], "a", "aa")], ("b", "a", "aa"))
    assert model_id_match(
        [(["bb", "b"], "a", "aa"), (["bb", "b"], "c", "cc")], ("bb", "a", "aa")
    )
