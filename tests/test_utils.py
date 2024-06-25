import pytest
import xarray as xr

from xmip.utils import (
    cmip6_dataset_id,
    google_cmip_col,
    instance_id_from_dataset,
    model_id_match,
)


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


def test_cmip6_dataset_id():
    ds = xr.Dataset({"data": 4})

    ds.attrs = {
        "activity_id": "ai",
        "institution_id": "ii",
        "source_id": "si",
        "variant_label": "vl",
        "experiment_id": "ei",
        "table_id": "ti",
        "grid_label": "gl",
        "variable_id": "vi",
    }

    assert cmip6_dataset_id(ds) == "ai.ii.si.ei.vl.ti.gl.none.vi"
    assert cmip6_dataset_id(ds, sep="_") == "ai_ii_si_ei_vl_ti_gl_none_vi"
    assert (
        cmip6_dataset_id(ds, id_attrs=["grid_label", "activity_id", "wrong_attrs"])
        == "gl.ai.none"
    )


class Test_instance_id_from_dataset:
    def test_default_cmip6(self):
        ds = xr.Dataset(
            attrs={
                "mip_era": "a",
                "grid_label": "b",
                "version": "c",
                "activity_id": "d",
                "institution_id": "e",
                "source_id": "f",
                "experiment_id": "g",
                "member_id": "h",
                "table_id": "i",
                "variable_id": "j",
            }
        )
        assert instance_id_from_dataset(ds) == "a.d.e.f.g.h.i.j.b.c"

    def test_custom_schema(self):
        ds = xr.Dataset(attrs={"some": "thing", "totally": "unrelated"})
        assert (
            instance_id_from_dataset(ds, id_schema="some.totally") == "thing.unrelated"
        )

    @pytest.mark.parametrize("missing_value", ["none", "some"])
    def test_missing_attrs_print_missing(self, missing_value):
        ds = xr.Dataset(
            attrs={
                "a": "a",
                "b": "b",
            }
        )
        iid = instance_id_from_dataset(
            ds, id_schema="a.b.c", print_missing=True, missing_value=missing_value
        )
        assert iid == f"a.b.{missing_value}"

    def test_missing_attrs_omit(self):
        ds = xr.Dataset(
            attrs={
                "a": "a",
                "b": "b",
            }
        )
        iid_omit = instance_id_from_dataset(ds, id_schema="a.b.c", print_missing=False)
        assert iid_omit == "a.b"

    def test_missing_attrs_warning(self):
        ds = xr.Dataset(
            attrs={
                "mip_era": "a",
                "activity_id": "d",
                "institution_id": "e",
                "source_id": "f",
                "experiment_id": "g",
                "member_id": "h",
                "table_id": "i",
                "variable_id": "j",
            }
        )
        with pytest.warns(
            UserWarning,
            match=r"Could not find dataset attributes for facets: \['grid_label', 'version'\]",
        ):
            instance_id_from_dataset(ds)
