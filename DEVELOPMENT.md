## Install BentoML from source

1. Make sure to have [Git](https://git-scm.com/), [pip](https://pip.pypa.io/en/stable/installation/), and [Python3.7+](https://www.python.org/downloads/) installed.

Optionally, make sure to have [GNU Make](https://www.gnu.org/software/make/) available on your system if you aren't using a UNIX-based system for a better developer experience.
If you don't want to use `make` then please refer to [Makefile](./Makefile) for specific commands on a given make target.

2. Clone the source code from BentoML's GitHub repository:
```bash
git clone https://github.com/bentoml/BentoML.git && cd BentoML
```

3. Install BentoML with pip in `editable` mode:
```bash
pip install -e .
```

This installs BentoML in an editable state. The changes you make will automatically be reflected even without reinstalling BentoML.

4. Test the BentoML installation either with `bash` or in an IPython session:
```bash
bentoml --version
```

```python
print(bentoml.__version__)
```

<details>
<summary><h2>Start Developing with VSCode</h2></summary>

1. Confirm that you have the following installed:
	- [Python3.7+](https://www.python.org/downloads/)
	- VSCode with the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) extensions

2. Clone the Github repository with the following steps:
	1. Open the command palette with Ctrl+Shift+P and type in clone.
	2. Select Git: Clone(Recursive).
	3. Clone BentoML.

3. Open a new terminal by clicking the Terminal dropdown at the top of the window, followed by the New Terminal option. Next, add a virtual environment with this command:
```bash
python -m venv .venv
```
4. Click yes if a popup suggests switching to the virtual environment. Otherwise, go through these steps:
	1. Open any python file in the directory.
	2. Select the interpreter selector on the blue status bar at the bottom of the editor.
  <img src="/docs/source/_static/img/vscode-status-bar.png" alt="VSCode Status Bar"></img>
	3. Switch to the path that includes .venv from the dropdown at the top.
  <img src="/docs/source/_static/img/vscode-select-venv.png" alt="VSCode Interpreter Selection Menu"></img>

5. Update your PowerShell execution policies. Win+x followed by the 'a' key opens the admin Windows PowerShell. Enter the following command to allow the virtual environment activation script to run:
```
	Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
</details>

## Testing

Make sure to install all test dependencies:
```bash
pip install -r requirements/tests-requirements.txt
```

If you are adding new frameworks it is recommended that you also added tests for our CI. Currently we are using GitHub Actions to manage our CI/CD workflow.

We recommended you to use [`nektos/act`](https://github.com/nektos/act) to run and tests Actions locally.


We introduce a tests script [run_tests.sh](./scripts/ci/run_tests.sh) that can be used to run tests locally and on CI.
```bash
./scripts/ci/run_tests.sh -h
Running unit/integration tests with pytest and generate coverage reports. Make sure that given testcases is defined under ./scripts/ci/config.yml.

Usage:
  ./scripts/ci/run_tests.sh [-h|--help] [-v|--verbose] <target> <pytest_additional_arguments>

Flags:
  -h, --help            show this message
  -v, --verbose         set verbose scripts


If pytest_additional_arguments is given, this will be appended to given tests run.

Example:
  $ ./scripts/ci/run_tests.sh pytorch --gpus --capture=tee-sys
```

All tests are then defined under [config.yml](./scripts/ci/config.yml) where each fields follow the following format:
```yaml
<target>: &tmpl
  root_test_dir: "tests/integration/frameworks"
  is_dir: false
  override_name_or_path:
  dependencies: []
  external_scripts:
  type_tests: "integration"
```

By default, each of our frameworks tests file with have the format: `test_<frameworks>_impl.py`. If `is_dir` set to `true` we will try to match the given `<target>` under `root_test_dir` to run tests from.

| Keys | Type | Defintions |
|------|------|------------|
|`root_test_dir`| `<str>`| root directory to run a given tests |
|`is_dir`| `<bool>`| whether `target` is a directory instead of a file |
|`override_name_or_path`| `<str>`| optional way to override a tests file name if doesn't match our convention |
|`dependencies`| `<List[str]>`| define additional dependencies required to run the tests, accepts `requirements.txt` format |
|`external_scripts`| `<str>`| optional shell scripts that can be run on top of `./scripts/ci/run_tests.sh` for given testsuite |
|`type_tests`| `<Literal["e2e","unit","integration"]>`| define type of tests for given `target` |

When `type_tests` is set to `e2e`, `./scripts/ci/run_tests.sh` will change current directory into given `root_test_dir` and will run testsuite from there.

The reason why we encourage developers to use the scripts in CI is that under the hood when we uses pytest, we will create a custom reports for given tests. This report then
 can be used as carryforward flags on codecov for consistent reporting.

Example:
```yaml
# e2e tests
general_features:
  root_test_dir: "tests/e2e/bento_server_general_features"
  is_dir: true
  type_tests: "e2e"
  dependencies:
    - "Pillow"

# framework
pytorch_lightning:
  <<: *tmpl
  dependencies:
    - "pytorch-lightning"
    - "-f https://download.pytorch.org/whl/torch_stable.html"
    - "torch==1.9.0+cpu"
    - "torchvision==0.10.0+cpu"
```

Refers to [config.yml](./scripts/ci/config.yml) for more examples.

### Unit tests

You can do this in two ways:

Run all unit tests directly with pytest:
```bash
# GIT_ROOT=$(git rev-parse --show-toplevel)
pytest tests/unit --cov=bentoml --cov-config="$GIT_ROOT"/setup.cfg
```

Run all unit tests via `./scripts/ci/run_tests.sh`:
```bash
./scripts/ci/run_tests.sh unit

# on UNIX-based system
make tests-unit
```

### Integration tests

Run given tests after defining target under `scripts/ci/config.yml` with `run_tests.sh`:
```bash
# example: run Keras TF1 integration tests
./scripts/ci/run_tests.sh keras_tf1
```

### E2E tests

```bash
# example: run e2e tests to check for general features
./scripts/ci/run_tests.sh general_features
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

## Style check, auto-formatting, type-checking

formatter: [black](https://github.com/psf/black), [isort](https://github.com/PyCQA/isort)

linter: [pylint](https://pylint.org/)

type checker: [pyright](https://github.com/microsoft/pyright)

Run linter/format script:
```bash
make format

make lint
```

Run type checker:
```bash
make type
```

## Documentations

Refers to [BentoML Documentation](./docs/README.md) for more information

Install all docs dependencies:
```bash
pip install -r requirements/docs-requirements.txt
```

To build documentation for locally:
```bash
cd docs/
make clean && make html
```


Modify `*.rst` files inside the `docs` folder to update content, and to
view your changes, run the following command:

```bash
python -m http.server --directory ./docs/build/html
```

Docs can then be accessed at [localhost:8000](http://localhost:8000)

If you are developing under macOS or Linux, we also made a script that watches docs
file changes, automatically rebuild the docs, and refreshes the browser
tab to show the change (UNIX-based system only):
```bash
./scripts/watch_docs.sh
```

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

#### Debian-based distros
Make sure you have `inotifywait` installed
```shell script
sudo apt install inotify-tools
```

## Python tools ecosystem

Currently BentoML are [PEP518](https://www.python.org/dev/peps/pep-0518/) compatible via `setup.cfg` and `pyproject.toml`.
 We also define most of our config for Python tools where:
 - `pyproject.toml` contains config for `setuptools`, `black`, `pytest`, `pylint`, `isort`, `pyright`
 - `setup.cfg` contains metadata for `bentoml` library and `coverage`

## Benchmark
BentoML has moved its benchmark to [`bentoml/benchmark`](https://github.com/bentoml/benchmark).

## Optional: git hooks

BentoML also provides git hooks that developers can install with:
```bash
make hooks
```
## Stubs
Refers to [Installation](https://github.com/microsoft/pyright#installation) to install pyright correctly.

In order to make pyright function correctly one also need to run the following scripts alongside with the stubs provided
 in the main repository.

One can also clone a [copy](https://github.com/bentoml/stubs) of all dependencies stubs used by BentoML to `typings/` via:
```bash
# Assuming at $GIT_ROOT
git clone git@github.com:bentoml/stubs.git
\rm -rf typings/.git* typings/*.{sh,toml,txt,cfg,py,md}
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
