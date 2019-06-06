## How to build BentoML locally

1. Pull the source code to local directory:
```bash
$ git pull https://github.com/bentoml/BentoML.git
$ cd BentoML
```

2. [Fork BentoML project](https://github.com/bentoml/BentoML/fork) on github and add upstream to local repository
```bash
$ git remote add upstream git@github.com:YOUR_USER_NAME/BentoML.git
```

3. Ensure you have python and pip installed, BentoML supports python _2.7_, _3.4_, _3.6_, and _3.7_
```bash
$ python --version
```
```bash
$ pip --version
```

4. Install all development and test dependencies:
```bash
pip install .[all]
```

5. Build and install BentoML with local branch:
```bash
$ pip install .
```

Now you should have BentoML installed:
```bash
$ bentoml --version
```


## How to run BentoML tests

1. Install all test dependencies:
```bash
pip install .[test]
```

2. Run all tests with current python version and environment
```bash
$ pytest
```

3. Run test under all supported python versions using Conda

Make sure you [have conda installed](https://docs.conda.io/projects/conda/en/latest/user-guide/install/):
```bash
$ conda --version
```

Bentoml tox file is configured to run in muiltple python versions:
```bash
$ tox
```

If you want to run tests under conda for specific version, use `-e` option:
```bash
$ tox -e py2.7
// or
$ tox -e py3.6
```

## Style check and auto-formatting your code

Make sure to install all dev dependencies:
```bash
$ pip install -e .[dev]

# For zsh users, use:
$ pip install -e .\[dev\]
```

Run linter/format script:
```bash
./script/format.sh

./script/lint.sh
```
