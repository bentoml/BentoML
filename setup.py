import os
import sys
import imp
import setuptools

__version__ = imp.load_source(
        'bentoml.version', os.path.join('bentoml', 'version.py')).__version__

with open("README.md", "r") as f:
    long_description = f.read()

install_requires = [
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
    'Werkzeug',
    'pathlib2',
    'requests',
    'packaging'
]

dev_requires = [
    'tox==3.8.4',
    'yapf==0.26.0',
    'pylint==2.3.1',
    'pytest==4.4.0',
    'tox-conda==0.2.0',
    'twine',
    'setuptools',
    'pycodestyle',
    'gitpython>=2.0.2'
]

cv2 = [ 'opencv-python' ]
pytorch  = [ 'torch', 'torchvision' ]
tensorflow = [ 'tensorflow' ]
gunicorn = [ 'gunicorn' ]

optional_requires = cv2 + pytorch + tensorflow + gunicorn
dev_all = install_requires + dev_requires + optional_requires

tests_require = [
    'pytest==4.4.0',
    'snapshottest==0.5.0',
    'mock==2.0.0',
    'tox==3.8.4'
    # 'pytest-cov',
    # 'coverage',
    # 'codecov'
]
tests_require += cv2

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
        'all': dev_all,
        'dev': dev_requires,
        'pytorch': pytorch,
        'tensorflow': tensorflow,
        'cv2': cv2,
        'test': tests_require,
    },
    url="https://github.com/bentoml/BentoML",
    packages=setuptools.find_packages(exclude=['tests*']),
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
        'Bug Reports': 'https://github.com/bentoml/BentoML/issues',
        'Source Code': 'https://github.com/bentoml/BentoML'
    }
)
