import setuptools

import versioneer

with open("README.md", "r", encoding="utf8") as f:
    long_description = f.read()

install_requires = [
    "aiohttp",
    "aiohttp_cors==0.7.0",
    "urllib3<=1.25.11",
    "cerberus",  # data validation
    "click>=7.0",
    "deepmerge",
    "docker",
    "humanfriendly",
    "numpy",
    "packaging",
    "psutil",
    # python-dateutil required by pandas and boto3, this makes sure the version
    # works for both
    "python-dateutil>=2.7.3,<3.0.0",
    "requests",
    "ruamel.yaml>=0.15.0",
    "schema",
    "tabulate",
    'contextvars;python_version < "3.7"',
    'dataclasses;python_version < "3.7"',
    "chardet",
    "simple-di==0.1.1",
    "cloudpickle",
    "mypy",
]


model_server_requires = [
    "opentracing",
    "py_zipkin",
    "jaeger_client",
    "prometheus_client",
    "python-json-logger",
]

test_requires = [
    "idna<=2.8",  # for moto
    "ecdsa==0.14",  # for moto
    "black==21.4b2",
    "codecov",
    "coverage>=4.4",
    "flake8>=3.8.2",
    "imageio>=2.5.0",
    "mock>=2.0.0",
    "moto==1.3.14",
    "pandas",
    "pylint>=2.9.3",
    "pytest-cov>=2.7.1",
    "pytest>=5.4.0",
    "pytest-asyncio",
    "parameterized",
    "scikit-learn",
    "isort>=5.0.0",
]

dev_requires = [
    "boto3",
    "flake8>=3.8.2",
    "gitpython>=2.0.2",
    "pylint>=2.5.2",
    "setuptools",
    "mypy",
    "autoflake",
    "tox-conda>=0.2.0",
    "tox>=3.12.1",
    "twine",
] + test_requires

docs_requires = [
    "recommonmark",
    "sphinx<=3.5.4",
    "sphinx-click",
    "sphinx_rtd_theme",
    "sphinxcontrib-fulltoc",
    "sphinxcontrib-spelling",
    "sphinx_copybutton",
    "pyenchant",
]

types_requires = [
    "types-click",
    "types-chardet",
    "types-setuptools",
    "pyspark-stubs",
]

dev_all = install_requires + dev_requires + docs_requires + types_requires

extras_require = {
    "dev": dev_all,
    "test": test_requires,
    "model_server": model_server_requires,
    "doc_builder": docs_requires,  # 'doc_builder' is required by readthedocs.io
}

setuptools.setup(
    name="BentoML",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="bentoml.org",
    author_email="contact@bentoml.ai",
    description="A framework for machine learning model serving",
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
    entry_points={"console_scripts": ["bentoml=bentoml:bentoml._internal.cli.cli"]},
    project_urls={
        "Bug Reports": "https://github.com/bentoml/BentoML/issues",
        "BentoML User Slack Group": "https://bit.ly/2N5IpbB",
        "Source Code": "https://github.com/bentoml/BentoML",
    },
    include_package_data=True,  # Required for '.cfg' files under bentoml/config
)
