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
$ tox -e py27
// or
$ tox -e py36
```

## Using forks/branches of BentoML

When trying new BentoML feature that has not been released, testing a fork of
BentoML on Google Colab or trying out changes in a pull request, an easy  way of
doing so is to use `pip install git+...` command, for example to install BentoML
from its master branch with all latest changes:

```
pip install git+https://github.com/bentoml/BentoML.git
```

Or to install from your own fork of BentoML:
```
pip install git+https://github.com/{your_github_username}/BentoML.git
```

You can also specify what branch to install from:
```
pip install git+https://github.com/{your_github_username}/BentoML.git@{branch_name}
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

## How to edit, run, build documentation site

Install all dev dependencies:
```bash
$ pip install -e .[dev]
```

To build documentation for locally:
```bash
$ ./script/build-docs.sh
```

Modify *.rst files inside the `docs` folder to update content, and to
view your changes, run the following command:

```
$ python -m http.server --directory built-docs
```

And go to your browser at `http://localhost:8000`
