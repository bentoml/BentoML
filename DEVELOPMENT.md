## Install BentoML from source

Ensure you have git, python and pip installed, BentoML supports Python 3.7+

```bash
python --version
```

```bash
pip --version
```


Clone the source code from BentoML's GitHub repository:
```bash
git clone https://github.com/bentoml/BentoML.git
cd BentoML
```

Install BentoML with pip in `editable` mode:
```bash
pip install --editable .
```

This will make `bentoml` available on your system which links to the sources of
your local clone and pick up changes you made locally.

Test the BentoML installation:
```bash
bentoml --version
```
```python
import bentoml
print(bentoml.__version__)
```

### Install BentoML from other forks or branches

The `pip` command support installing directly from remote git repository. This makes it
easy to try out new BentoML feature that has not been released, test changes in a pull 
request. For example, to install BentoML from its master branch:

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



## How to run unit tests

1. Install all test dependencies:
```
pip install -e ".[test]"
```

2. Run all unit tests with current python version and environment
```bash
./ci/unit_tests.sh
```

## How to run integration tests

After isntall all test dependencies, to run a specific integration tests suite after adding testcases do:
```bash
# for example you added tests for mlflow
pytest tests/integration/frameworks/mlflow --cov=bentoml --cov-config=.coveragerc
```

### Optional: Run unit test with all supported python versions

Make sure you [have conda installed](https://docs.conda.io/projects/conda/en/latest/user-guide/install/):
```bash
conda --version
```

Bentoml tox file is configured to run in muiltple python versions:
```bash
tox
```

If you want to run tests under conda for specific version, use `-e` option:
```bash
tox -e py37
// or
tox -e py36
```

## Run BentoML with verbose/debug logging

Add the following lines to the Python code that invokes BentoML:

```python
from bentoml.configuration import set_debug_mode
set_debug_mode(True)
```

And/or set the `BENTOML_DEBUG` environment variable to `TRUE`:
```bash
export BENTOML_DEBUG=TRUE
```

And/or use the `--verbose` option when running `bentoml` CLI command, e.g.:
```bash
bentoml get IrisClassifier --verbose
```

## Style check and auto-formatting your code

Make sure to install all dev dependencies:
```bash
pip install -e ".[dev]"
```

Run linter/format script:
```bash
# if you have GNU make available you can do `make format`
./dev/format.sh

# if you have GNU make available you can do `make lint`
./dev/lint.sh
```

### Optional: Running `mypy` for better type annotation

Make sure to install [mypy](https://mypy.readthedocs.io/en/stable/getting_started.html)

You might have to install stubs before running:
```bash
mypy --install-types
```

After updating/modifying codebase (e.g: `bentoml/pytorch.py`), run `mypy`:
```bash
mypy bentoml/pytorch.py
```

## How to edit, run, build documentation site

Install all dev dependencies:
```bash
pip install -e ".[dev]"
```

To build documentation for locally:
```bash
./docs/build.sh
```

Modify `*.rst` files inside the `docs` folder to update content, and to
view your changes, run the following command:

```
python -m http.server --directory ./docs/build/html
```

And go to your browser at `http://localhost:8000`

If you are developing under macOS or linux, we also made a script that watches docs
file changes, automatically rebuild the docs, and refreshes the browser
tab to show the change (macOS only):

## How to run spell check for documentation site

Install all dev dependencies:
```bash
pip install -e ".[dev]"
```

Install spellchecker dependencies:
```bash
make install-spellchecker-deps
```

To run spellchecker locally:
```bash
make spellcheck-doc
```

### macOS

Make sure you have fswatch command installed:
```
brew install fswatch
```

Run the `watch.sh` script to start watching docs changes:
```bash
./docs/watch.sh
```

### Linux
Make sure you have `inotifywait` installed
```shell script
sudo apt install inotify-tools
``` 

Run the `watch.sh` script to start watching docs changes:
```bash
./docs/watch.sh
```

## Running BentoML Benchmark
BentoML has moved its benchmark to [`bentoml/benchmark`](https://github.com/bentoml/benchmark).

## How to test GitHub Actions locally

If you are developing new artifacts or modify GitHub Actions CI (adding integration test, unit tests, etc), use [`nektos/act`](https://github.com/nektos/act) to run Actions locally.


## Creating Pull Request on GitHub

### Optional[RECOMMENDED]: Install Git hooks

Run `./dev/install_git_hooks.sh` to install git hooks to automate
commit and branch format enforcement described above.

1. [Fork BentoML project](https://github.com/bentoml/BentoML/fork) on GitHub and
add upstream remotes to local BentoML clone:

```bash
git remote add upstream git@github.com:bentoml/BentoML.git
```

2. Make the changes either to fix a known issue or adding new feature

3. In order for us to manage PR and Issues systematically, we encourage developers to use hierarchical branch folders to manage branch naming.
   Run `./dev/install_git_hooks.sh` to install `pre-commit` hooks. We will check if your branch naming
   follows the given regex : `^(feature|bugfix|improv|lib|prerelease|release|hotfix)\/[a-zA-Z0-9._-]+$`. This
   is partially based on how [Azure DevOps](https://docs.microsoft.com/en-us/azure/devops/repos/git/require-branch-folders?view=azure-devops&tabs=browser)
   manages its repos and conventional commits spec (see below for more information):
   - feature: new features/proposals users want to integrate into the library, not a new feature for a build script
   - bugfix: bugfix of a feature, not a fix to a build script
   - improv: improvements/refactor/cleanup production code, eg. reformat, pylint, etc.
   - lib: related to internal libraries, features required by the production code
   - prerelease: alpha/beta features that might should be included in the prerelease of the library. This would help testing new features/integrations for the library
   - release: included all features that is production ready
   - hotifx: patch of current bugs in production code

4. Push changes to your fork and follow [this
   article](https://help.github.com/en/articles/creating-a-pull-request)
   on how to create a pull request on github. Name your pull request
   with one of the following prefixes, e.g. "feat: add support for
   PyTorch". This is based on the [Conventional Commits
   specification](https://www.conventionalcommits.org/en/v1.0.0/#summary)
   - feat: (new feature for the user, not a new feature for build script)
   - fix: (bug fix for the user, not a fix to a build script)
   - docs: (changes to the documentation)
   - style: (formatting, missing semicolons, etc; no production code change)
   - refactor: (refactoring production code, eg. renaming a variable)
   - perf: (code changes that improve performance)
   - test: (adding missing tests, refactoring tests; no production code change)
   - chore: (updating grunt tasks etc; no production code change)
   - build: (changes that affect the build system or external dependencies)
   - ci: (changes to configuration files and scripts)
   - revert: (reverts a previous commit)

5. Once your pull request created, an automated test run will be triggered on
   your branch and the BentoML authors will be notified to review your code
   changes. Once tests are passed and reviewer has signed off, we will merge
   your pull request.

