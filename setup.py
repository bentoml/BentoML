import setuptools

__version__ = None
exec(open('bentoml/version.py').read()) # load and overwrite __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    'pathlib2',  # TODO: only install this when python version < (3.4)
    'prometheus_client',
    'ruamel.yaml>=0.15.0',
    'numpy',
    'flask',
    'gunicorn',
    'six',
    'click',
    'pandas',
    'dill',
    'python-json-logger',
    'boto3',
]

optional_requires = [
    'torch',
    'torchvision'
]

dev_requires = [
    'tox==3.8.4',
    'yapf==0.26.0',
    'pylint==2.3.1',
    'pytest==4.4.0',
    'tox-conda==0.2.0',
    'twine',
    'setuptools',
    'pycodestyle'
]

test_requires = [
    'pytest==4.4.0',
    'snapshottest==0.5.0',
    'mock==2.0.0',
    'tox==3.8.4'
    # 'pytest-cov',
    # 'coverage',
    # 'codecov'
]

setuptools.setup(
    name="BentoML",
    version=__version__,
    author="atalaya.io",
    author_email="contact@atalaya.io",
    description="BentoML: Package and Deploy Your Machine Learning Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    extras_require={
        'optional': optional_requires,
        'dev': dev_requires,
        'test': test_requires,
    },
    url="https://github.com/atalaya-io/BentoML",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent"
    ],
    python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*",
    entry_points={
        'console_scripts': [
            'bentoml=bentoml.cli:cli',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/atalaya-io/bentoml/issues',
        'Source Code': 'https://github.com/atalaya-io/bentoml'
    }
)
