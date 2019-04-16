import pandas as pd


def test_pip_install_bento_archive(bento_archive_path):
    import subprocess
    subprocess.call(['pip', 'install', bento_archive_path])

    test_bento_service_module = __import__('TestBentoService')
    svc = test_bento_service_module.load()
    df = svc.predict(pd.DataFrame(pd.DataFrame([1], columns=['age'])))
    assert df['age'].values[0] == 6
    subprocess.call(['pip', 'uninstall', 'TestBentoService'])
