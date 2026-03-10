import warnings


def test_importing_legacy_module_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        import adeptly.dqn  # noqa: F401

    assert any(issubclass(w.category, DeprecationWarning) for w in captured)
