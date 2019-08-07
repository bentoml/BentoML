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
    "six",
    "click",
    "pandas",
    "dill",
    "prometheus_client",
    "python-json-logger",
    "boto3",
    "pathlib2",
    "requests",
    "packaging",
    "docker",
    "configparser",
    "sqlalchemy",
    "protobuf>=3.6.0",
]

imageio = ["imageio>=2.5.0"]
cv2 = ["opencv-python"]
pytorch = ["torch", "torchvision"]
fastai = ["fastai", "matplotlib"]
tensorflow = ["tensorflow"]
xgboost = ["xgboost"]
h2o = ["h2o"]
api_server = ["gunicorn", "prometheus_client", "Werkzeug"]
grpc_tools = ["grpcio", "grpcio-tools", "mypy-protobuf"]

optional_requires = api_server + imageio + pytorch + tensorflow + fastai + xgboost + h2o

tests_require = (
    [
        "pytest==4.1.0",
        "pytest-cov==2.7.1",
        "snapshottest==0.5.0",
        "mock==2.0.0",
        "tox==3.12.1",
        "coverage>=4.4",
        "codecov",
    ]
    + imageio
    + cv2
    + fastai
)

dev_requires = (
    [
        "pylint==2.3.1",
        "flake8",
        "tox-conda==0.2.0",
        "twine",
        "black",
        "setuptools",
        "gitpython>=2.0.2",
    ]
    + tests_require
    + grpc_tools
)

sphinx_requires = [
    "sphinx",
    "sphinx-click",
    "sphinx_rtd_theme",
    "sphinxcontrib-fulltoc",
]

doc_builder_requires = sphinx_requires + install_requires

dev_all = (
    install_requires + dev_requires + optional_requires + sphinx_requires + grpc_tools
)

extras_require = {
    "all": dev_all,
    "api_server": api_server,
    "dev": dev_requires,
    "doc_builder": doc_builder_requires,
    "pytorch": pytorch,
    "tensorflow": tensorflow,
    "imageio": imageio,
    "test": tests_require,
    "fastai": fastai,
    "xgboost": xgboost,
    "h2o": h2o,
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
        "Programming Language :: Python :: 3.5",
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
