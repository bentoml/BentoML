import sys
import pandas as pd


def test_pip_install_bento_archive(bento_archive_path, tmpdir):
    install_path = str(tmpdir.mkdir('pip_local'))
    import subprocess
    output = subprocess.check_output(
        ['pip', 'install', '--target={}'.format(install_path), bento_archive_path]).decode()
    assert 'Successfully installed TestBentoService' in output

    sys.path.insert(0, install_path)
    import TestBentoService
    sys.path.remove(install_path)

    svc = TestBentoService.load()
    df = svc.predict(pd.DataFrame(pd.DataFrame([1], columns=['age'])))
    assert df['age'].values[0] == 6
