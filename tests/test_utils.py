import pytest

from cmip6_preprocessing.utils import google_cmip_col, model_id_match


def test_google_cmip_col():
    try:
        import intake
    except ImportError:
        intake = None
    if intake is None:
        with pytest.raises(ImportError):
            col = google_cmip_col(catalog="main")
    else:
        col = google_cmip_col(catalog="main")
        assert (
            col.catalog_file == "https://storage.googleapis.com/cmip6/pangeo-cmip6.csv"
        )

        with pytest.raises(ValueError):
            col = google_cmip_col(catalog="wrong")


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
