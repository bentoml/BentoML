## Install BentoML from source

Make sure to have [Git](https://git-scm.com/) and [Python3.7+](https://www.python.org/downloads/) installed.

Optionally, make sure to have [GNU Make](https://www.gnu.org/software/make/) available on your system if you aren't using UNIX-based system for better developer experience.
If you don't want to use `make` then refers to [Makefile](./Makefile) for specific commands on given make target.

We are also using [docker](https://www.docker.com/) for our code style scripts that will be mentioned below for better devex.

```bash
python --version

pip --version
```

Clone the source code from BentoML's GitHub repository:
```bash
git clone --recurse-submodules https://github.com/bentoml/BentoML.git && cd BentoML
```

If you want to use [`poetry`](https://python-poetry.org/) then do:

```bash
poetry install
```

Install BentoML with pip in `editable` mode:
```bash
make install-local
```

This will make `bentoml` available on your system which links to the sources of
your local clone and pick up changes you made locally.

Test the BentoML installation either with `bash` or in an IPython session:
```bash
bentoml --version
```

```python
print(bentoml.__version__)
```

### Install BentoML from other forks or branches

`pip` also supports installing directly from remote git repository. This makes it
easy to try out new BentoML feature that has not been released, test changes in a pull
request. For example, to install BentoML from its master branch:

```bash
pip install git+https://github.com/bentoml/BentoML.git
```

Or to install from your own fork of BentoML:
```bash
pip install git+https://github.com/{your_github_username}/BentoML.git
```

You can also specify what branch to install from:
```bash
pip install git+https://github.com/{your_github_username}/BentoML.git@{branch_name}
```


## Testing

Make sure to install all test dependencies:
```bash
make install-dev-deps
```

### Unit tests

Run all unit tests with current python version and environment
```bash
./scripts/ci/unit_tests.sh
```

#### Optional: Run unit test with all supported python versions

Make sure you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed:

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

### Integration tests

After install all test dependencies, to run a specific integration tests suite after adding testcases do:

```bash
# `make integration-tests-<frameworks>` will trigger integration tests for that specific frameworks.
#  Make sure that your frameworks is defined under ./scripts/ci/config.yml

make integration-tests-mlflow
```

If you are adding new frameworks it is recommended that you also added tests for our CI. Currently we are using GitHub Actions to manage our CI/CD workflow.

We recommended you to use [`nektos/act`](https://github.com/nektos/act) to run and tests Actions locally.


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

## Style check, auto-formatting, type-checking

formatter: [black](https://github.com/psf/black), [isort](https://github.com/PyCQA/isort)

linter: [flake8](https://flake8.pycqa.org/en/latest/), [pylint](https://pylint.org/)

type checker: [mypy](https://mypy.readthedocs.io/en/stable/), [pyright](https://github.com/microsoft/pyright)

### [Required]: Docker

Run linter/format script:
```bash
make format

make lint
```

Run type checker:
```bash
make type
```

### Without Docker

Make sure to install all dev dependencies:
```bash
make install-dev-deps
```

Run linter/format script:
```bash
./scripts/tools/formatter.sh

./scripts/tools/linter.sh
```

Run type checker:
```bash
./scripts/tools/type_checker.sh
```

## Documentations

Refers to [BentoML Documentation](./docs/README.md) for more information

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

```bash
python -m http.server --directory ./docs/build/html
```

Docs can then be accessed at [localhost:8000](http://localhost:8000)

If you are developing under macOS or linux, we also made a script that watches docs
file changes, automatically rebuild the docs, and refreshes the browser
tab to show the change (macOS only):

### Running spellcheck for documentation site.

Install spellchecker dependencies:
```bash
make install-spellchecker-deps
```

To run spellchecker locally:
```bash
make spellcheck-doc
```

#### macOS

Make sure you have fswatch command installed:
```
brew install fswatch
```

Run the `watch.sh` script to start watching docs changes:
```bash
./scripts/watch_docs.sh
```

#### Debian-based distros
Make sure you have `inotifywait` installed
```shell script
sudo apt install inotify-tools
``` 

Run the `watch.sh` script to start watching docs changes:
```bash
./scripts/watch_docs.sh
```

## Benchmark
BentoML has moved its benchmark to [`bentoml/benchmark`](https://github.com/bentoml/benchmark).

## Optional: git hooks

BentoML also provides git hooks that developers can install with:
```bash
make hooks
```

## Creating Pull Request on GitHub

[Fork BentoML project](https://github.com/bentoml/BentoML/fork) on GitHub and
add upstream remotes to local BentoML clone:

```bash
git remote add upstream git@github.com:bentoml/BentoML.git
```

Make the changes either to fix a known issue or adding new feature

In order for us to manage PR and Issues systematically, we encourage developers to use hierarchical branch folders to manage branch naming.
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

Push changes to your fork and follow [this
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

Once your pull request created, an automated test run will be triggered on
your branch and the BentoML authors will be notified to review your code
changes. Once tests are passed and reviewer has signed off, we will merge
your pull request.

