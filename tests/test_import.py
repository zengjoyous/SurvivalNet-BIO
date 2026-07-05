def test_public_api_imports():
    import survivalnet

    assert survivalnet.__version__
    assert callable(survivalnet.load_table)
    assert callable(survivalnet.prepare_feature_matrix)
    assert callable(survivalnet.CoxModel)
