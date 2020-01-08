## Install BentoML from source code

Download the source code to local file system:
```bash
$ git clone https://github.com/bentoml/BentoML.git
$ cd BentoML
```

Ensure you have python and pip installed, BentoML supports python _3.6_, and _3.7_
```bash
$ python --version
```
```bash
$ pip --version
```

And install BentoML with pip in `editable` mode:
```
pip install --editable .
```

This will make `bentoml` available on your system which links to the sources of
your local clone and pick up changes you made locally.

Now you can test your BentoML installation by running the following in terminal:
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
$ tox -e py37
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
$ ./docs/build.sh
```

Modify \*.rst files inside the `docs` folder to update content, and to
view your changes, run the following command:

```
$ python -m http.server --directory ./docs/build/html
```

And go to your browser at `http://localhost:8000`

If you are developing under `macOS`, we also made a script that watches docs
file changes, automatically rebuild the docs html files, and refresh the browser
tab to show the change:

Make sure you have fswatch command installed:
```
brew install fswatch
```

Run the `watch.sh` script to start watching docs changes:
```
$ ./docs/watch.sh
```


## Creating Pull Request on Github


1. [Fork BentoML project](https://github.com/bentoml/BentoML/fork) on github and
add upstream to local BentoML clone:

```bash
$ git remote add upstream git@github.com:YOUR_USER_NAME/BentoML.git
```

2. Make the changes either to fix a known issue or adding new feature

3. Push changes to your fork and follow [this
   article](https://help.github.com/en/articles/creating-a-pull-request)
   on how to create a pull request on github

4. Once your pull request created, an automated test run will be triggered on
   your branch and the BentoML authors will be notified to review your code
   changes. Once tests are passed and reviewer has signed off, we will merge
   your pull request.
