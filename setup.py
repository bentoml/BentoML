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

with open("README.md", "r", encoding="utf8") as f:
    long_description = f.read()

install_requires = [
    "aiohttp",
    "alembic",
    "boto3",
    "cerberus",
    "certifi",
    "click>=7.0",
    "configparser",
    "docker",
    "flask",
    "grpcio<=1.27.2",
    "gunicorn",
    "humanfriendly",
    "numpy",
    "packaging",
    "prometheus_client",
    "protobuf>=3.6.0",
    "psutil",
    "py_zipkin",
    # python-dateutil required by pandas and boto3, this makes sure the version
    # works for both
    "python-dateutil>=2.1,<2.8.1",
    "python-json-logger",
    "requests",
    "ruamel.yaml>=0.15.0",
    "sqlalchemy-utils",
    "sqlalchemy>=1.3.0",
    "tabulate",
    'contextvars;python_version < "3.7"',
    "multidict",
]

aws_sam_cli = ["aws-sam-cli==0.33.1"]
azure_cli = ["azure-cli"]
postgres = ['psycopg2', 'psycopg2-binary']

test_requires = [
    "black==19.10b0",
    "codecov",
    "coverage>=4.4",
    "flake8>=3.8.2",
    "imageio>=2.5.0",
    "mock>=2.0.0",
    "moto",
    "pandas",
    "pylint>=2.5.2",
    "pytest-cov>=2.7.1",
    "pytest>=5.4.0",
    "pytest-asyncio",
    "scikit-learn",
    "protobuf==3.6.0",
] + aws_sam_cli

dev_requires = [
    "flake8>=3.8.2",
    "gitpython>=2.0.2",
    "grpcio-reflection<=1.27.2",
    # This grpcio-tools version  should be kept in sync with the version found in
    # `protos/generate-docker.sh` script
    "grpcio-tools==1.27.2",
    "pylint>=2.5.2",
    "setuptools",
    "tox-conda>=0.2.0",
    "tox>=3.12.1",
    "twine",
] + test_requires

docs_requires = [
    "recommonmark",
    "sphinx",
    "sphinx-click",
    "sphinx_rtd_theme",
    "sphinxcontrib-fulltoc",
]

dev_all = install_requires + dev_requires + docs_requires

yatai_service = install_requires + aws_sam_cli + postgres + azure_cli

extras_require = {
    "dev": dev_all,
    "doc_builder": docs_requires + install_requires,  # required by readthedocs.io
    "test": test_requires,
    "yatai_service": yatai_service,
}

setuptools.setup(
    name="BentoML",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="bentoml.org",
    author_email="contact@bentoml.ai",
    description="An open-source ML model serving and model management framework",
    long_description=long_description,
    license="Apache License 2.0",
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    extras_require=extras_require,
    url="https://github.com/bentoml/BentoML",
    packages=setuptools.find_packages(exclude=["tests*"]),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6.1",
    entry_points={"console_scripts": ["bentoml=bentoml.cli:cli"]},
    project_urls={
        "Bug Reports": "https://github.com/bentoml/BentoML/issues",
        "BentoML User Slack Group": "https://bit.ly/2N5IpbB",
        "Source Code": "https://github.com/bentoml/BentoML",
    },
    include_package_data=True,  # Required for '.cfg' files under bentoml/config
)
