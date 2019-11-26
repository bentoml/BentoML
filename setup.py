# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import setuptools
import versioneer

PY3 = sys.version_info.major == 3

with open("README.md", "r") as f:
    long_description = f.read()

install_requires = [
    "ruamel.yaml>=0.15.0",
    "numpy",
    "flask",
    "gunicorn",
    "six",
    "click>=7.0",
    "pandas",
    "prometheus_client",
    "python-json-logger",
    "boto3",
    "pathlib2",
    "requests",
    "packaging",
    "docker",
    "configparser",
    "sqlalchemy>=1.3.0",
    "protobuf>=3.6.0",
    "grpcio",
    "cerberus",
    "tabulate",
    "humanfriendly",
    "alembic",
]

imageio = ["imageio>=2.5.0"]
pytorch = ["torch", "torchvision"]
fastai = ["fastai", "matplotlib"]
tensorflow = ["tensorflow"]
xgboost = ["xgboost"]
h2o = ["h2o"]
api_server = ["gunicorn", "prometheus_client"]
aws_sam_cli = ["aws-sam-cli"]

optional_requires = (
    api_server + imageio + pytorch + tensorflow + fastai + xgboost + h2o + aws_sam_cli
)

test_requires = (
    [
        "pytest>=4.1.0",
        "pytest-cov>=2.7.1",
        "snapshottest>=0.5.0",
        "mock>=2.0.0",
        "tox>=3.12.1",
        "coverage>=4.4",
        "codecov",
        "moto",
    ]
    + imageio
    + fastai
    + aws_sam_cli
)

dev_requires = [
    "flake8",
    "twine",
    "setuptools",
    "gitpython>=2.0.2",
    "grpcio-tools",
] + test_requires

if PY3:
    # Python3 only python dev tools
    dev_requires += ["pylint>=2.3.1", "tox-conda>=0.2.0", "black"]
else:
    dev_requires += ["pylint"]

docs_requires = ["sphinx", "sphinx-click", "sphinx_rtd_theme", "sphinxcontrib-fulltoc"]

dev_all = install_requires + dev_requires + optional_requires + docs_requires

extras_require = {
    "all": dev_all,
    "dev": dev_requires,
    "api_server": api_server,
    "test": test_requires,
    "doc_builder": docs_requires + install_requires,  # required by readthedocs.io
}

setuptools.setup(
    name="BentoML",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="atalaya.io",
    author_email="contact@atalaya.io",
    description="A python framework for serving and operating machine learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    extras_require=extras_require,
    url="https://github.com/bentoml/BentoML",
    packages=setuptools.find_packages(exclude=["tests*"]),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*",
    entry_points={"console_scripts": ["bentoml=bentoml.cli:cli"]},
    project_urls={
        "Bug Reports": "https://github.com/bentoml/BentoML/issues",
        "Source Code": "https://github.com/bentoml/BentoML",
        "Slack User Group": "https://bit.ly/2N5IpbB",
    },
    include_package_data=True,  # Required for '.cfg' files under bentoml/config
)
