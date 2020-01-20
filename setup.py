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

import setuptools
import versioneer

with open("README.md", "r") as f:
    long_description = f.read()

install_requires = [
    "ruamel.yaml>=0.15.0",
    "numpy",
    "flask",
    "gunicorn",
    "click>=7.0",
    "pandas",
    "prometheus_client",
    "python-json-logger",
    "boto3",
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
    # python-dateutil required by pandas and boto3, this makes sure the version
    # works for both
    "python-dateutil>=2.1,<2.8.1",
]

imageio = ["imageio>=2.5.0"]
pytorch = ["torch", "torchvision"]
fastai = ["fastai", "matplotlib"]
tensorflow = ["tensorflow"]
xgboost = ["xgboost"]
h2o = ["h2o"]
api_server = ["gunicorn", "prometheus_client"]
aws_sam_cli = ["aws-sam-cli==0.33.1"]

optional_requires = (
    api_server + imageio + pytorch + tensorflow + fastai + xgboost + h2o + aws_sam_cli
)

test_requires = (
    [
        "pytest>=4.1.0",
        "pytest-cov>=2.7.1",
        "mock>=2.0.0",
        "coverage>=4.4",
        "codecov",
        "moto",
        "numpy",
    ]
    + imageio
    + aws_sam_cli
)

dev_requires = [
    "tox>=3.12.1",
    "tox-conda>=0.2.0",
    "flake8",
    "twine",
    "setuptools",
    "gitpython>=2.0.2",
    "grpcio-tools",
    "pylint>=2.3.1",
    "black",
]


docs_requires = [
    "sphinx",
    "sphinx-click",
    "sphinx_rtd_theme",
    "sphinxcontrib-fulltoc",
    "recommonmark",
]

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
    author="bentoml.org",
    description="A platform for serving and deploying machine learning models in the "
    "cloud",
    long_description=long_description,
    license="Apache License 2.0",
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    extras_require=extras_require,
    url="https://github.com/bentoml/BentoML",
    packages=setuptools.find_packages(exclude=["tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.4",
    entry_points={"console_scripts": ["bentoml=bentoml.cli:cli"]},
    project_urls={
        "Bug Reports": "https://github.com/bentoml/BentoML/issues",
        "Source Code": "https://github.com/bentoml/BentoML",
        "Slack User Group": "https://bit.ly/2N5IpbB",
    },
    include_package_data=True,  # Required for '.cfg' files under bentoml/config
)
