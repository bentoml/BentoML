import requests
import os
import zipfile
from distlib.util import parse_requirement

manylinux_whell_file_suffix = {
    'python2.7': 'cp27mu-manylinux1_x86_64.whl',
    'python3.6': 'cp36m-manylinux1_x86_64.whl',
    'python3.7': 'cp37m-manylinux1_x86_64.whl'
}


def get_manylinux_wheel_url(python_version, package_name, package_version):
    """
    Return downloadable URL from pypi with given package info, else return None
    """
    url = 'https://pypi.python.org/pypi/{}/json'.format(package_name)
    res = requests.get(url, timeout=1.5)
    data = res.json()

    if package_version not in data['releases']:
        return None

    for f in data['releases'][package_version]:
        if f['filename'].endswith(manylinux_whell_file_suffix[python_version]):
            wheel_url = f['url']

    if package_name == 'bentoml':
        wheel_url = data['releases'][package_version][0]['url']

    return wheel_url


def download_manylinux_wheel_from_url(url, wheel_path):
    if not os.path.exists(wheel_path) or not zipfile.is_zipfile(wheel_path):
        with open(wheel_path, 'wb') as f:
            resp = requests.get(url, timeout=2, stream=True)
            resp.raw.decode_content = True
            for chunk in resp.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    return wheel_path


def read_requirment_txt(requirment_txt_path):
    with open(requirment_txt_path, 'r') as f:
        content = f.read()
    package_list = content.splitlines()
    parsed_package_list = {}
    for item in package_list:
        result = parse_requirement(item)
        parsed_package_list[result.name] = result
    return parsed_package_list
