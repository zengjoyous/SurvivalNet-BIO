def test_public_package_imports():
    import survivalnet

    assert hasattr(survivalnet, "__version__")
    assert callable(survivalnet.load_table)
    assert callable(survivalnet.prepare_feature_matrix)
