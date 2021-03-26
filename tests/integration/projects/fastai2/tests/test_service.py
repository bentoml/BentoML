import pandas

test_df = pandas.DataFrame([[1] * 5])


def test_fastai2_artifact_pack(service):
    assert service.predict(test_df) == 5.0, 'Run inference before saving'
