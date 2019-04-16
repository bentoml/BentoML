import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_pip_install_bento_archive(bento_archive_path, tmpdir):
    import subprocess

    install_path = str(tmpdir.mkdir('pip_local'))
    output = subprocess.check_output(
        ['pip', 'install', '--target={}'.format(install_path), bento_archive_path]).decode()
    assert 'Successfully installed TestBentoService' in output

    sys.path.append(install_path)
    TestBentoService = __import__('TestBentoService')
    sys.path.remove(install_path)

    svc = TestBentoService.load()
    df = svc.predict(pd.DataFrame(pd.DataFrame([1], columns=['age'])))
    assert df['age'].values[0] == 6
