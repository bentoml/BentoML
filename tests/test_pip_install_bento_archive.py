import os
import sys
import json

import pandas as pd


def test_pip_install_bento_archive(bento_archive_path, tmpdir):
    import subprocess

    install_path = str(tmpdir.mkdir('pip_local'))
    bentoml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    output = subprocess.check_output(
        ['pip', 'install', '--target={}'.format(install_path), bento_archive_path]).decode()
    assert 'Successfully installed TestBentoService' in output

    sys.path.append(install_path)
    TestBentoService = __import__('TestBentoService')
    sys.path.remove(install_path)

    svc = TestBentoService.load()
    df = svc.predict(pd.DataFrame(pd.DataFrame([1], columns=['age'])))
    assert df['age'].values[0] == 6

    # pip install should place cli entry script under target/bin directory
    cli_bin_path = os.path.join(install_path, 'bin/TestBentoService')

    # add install_path and local bentoml module to PYTHONPATH to make them
    # available in subprocess call
    env = {'PYTHONPATH': install_path + ':' + bentoml_path}

    output = subprocess.check_output([cli_bin_path, 'info'], env=env).decode()
    output = json.loads(output)
    assert output['name'] == 'TestBentoService'
    assert output['version'] == svc.version
    assert 'predict' in output['apis']
