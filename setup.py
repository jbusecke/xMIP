from setuptools import setup


setup(
    use_scm_version={
        "write_to": "cmip6_preprocessing/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    },
    setup_requires=["setuptools>=45", "setuptools_scm[toml]>=6.0"],
)
