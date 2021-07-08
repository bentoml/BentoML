## Install BentoML from source

Ensure you have git, python and pip installed, BentoML supports python 3.6, 3.7, and 3.8

```bash
$ python --version
```

```bash
$ pip --version
```


Download the source code from BentoML's Github repository:
```bash
$ git clone https://github.com/bentoml/BentoML.git
$ cd BentoML
```

Install BentoML with pip in `editable` mode:
```
pip install --editable .
```

This will make `bentoml` available on your system which links to the sources of
your local clone and pick up changes you made locally.

Test the BentoML installation:
```bash
$ bentoml --version
```
```python
import bentoml
print(bentoml.__version__)
```

#### Install BentoML from other forks or branches

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
$ pip install -e ".[test]"
```

2. Run all unit tests with current python version and environment
```bash
$ ./ci/unit_tests.sh
```

## Optional: Run unit test with all supported python versions

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
$ pip install -e ".[dev]"
```

Run linter/format script:
```bash
./dev/format.sh

./dev/lint.sh
```

## How to edit, run, build documentation site

Install all dev dependencies:
```bash
$ pip install -e ".[dev]"
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

If you are developing under macOS or linux, we also made a script that watches docs
file changes, automatically rebuild the docs, and refreshes the browser
tab to show the change (macOS only):

## How to run spell check for documentation site

Install all dev dependencies:
```bash
$ pip install -e ".[dev]"
```

Install spellchecker dependencies:
```
$ make install-spellchecker-deps
```

To run spellchecker locally:
```bash
$ make spellcheck-doc
```


### macOS

Make sure you have fswatch command installed:
```
brew install fswatch
```

Run the `watch.sh` script to start watching docs changes:
```
$ ./docs/watch.sh
```

### Linux
Make sure you have `inotifywait` installed
```shell script
sudo apt install inotify-tools
``` 

Run the `watch.sh` script to start watching docs changes:
```
$ ./docs/watch.sh
```

## How to debug YataiService GRPC server

Install all dev dependencies:
```bash
$ pip install -e ".[dev]"
```

Install grpcui:
```bash
$ go get github.com/fullstorydev/grpcui
$ go install github.com/fullstorydev/grpcui/cmd/grpcui
```

Start Yatai server in debug mode:
```bash
$ bentoml yatai-service-start --debug
```

In another terminal session run grpcui:
```bash
$ grpcui -plain text localhost:50051

gRPC Web UI available at http://127.0.0.1:60551/...
```
Navigate to the URL from above

## How to use `YataiService` helm chart

BentoML also provides a Helm chart under [`bentoml/yatai-chart`](https://github.com/bentoml/yatai-chart) for installing YataiService on Kubernetes.

## Running BentoML Benchmark
BentoML has moved its benchmark client to [`bentoml/benchmark`](https://github.com/bentoml/benchmark).

## How to run and develop BentoML Web UI

Make sure you have `yarn` installed: https://classic.yarnpkg.com/en/docs/install 

Install all npm packages required by BentoML Web UI:

```bash
# install npm packages required by BentoML's Node.js Web Server
cd {PROJECT_ROOT}/bentoml/yatai/web/
yarn

# install npm packages required by BentoML web frontend
cd {PROJECT_ROOT}/bentoml/yatai/web/client/
yarn
```

Build the Web Server and frontend UI code:
```bash
cd {PROJECT_ROOT}/bentoml/yatai/web/
npm run build
```

## How to test GitHub Actions locally

If you are developing new artifacts or modify GitHub Actions CI (adding integration test, unit tests, etc), use [`nektos/act`](https://github.com/nektos/act) to run Actions locally.

## Creating Pull Request on Github


1. [Fork BentoML project](https://github.com/bentoml/BentoML/fork) on github and
add upstream to local BentoML clone:

```bash
$ git remote add upstream git@github.com:YOUR_USER_NAME/BentoML.git
```

2. Make the changes either to fix a known issue or adding new feature

3. Push changes to your fork and follow [this
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

4. Once your pull request created, an automated test run will be triggered on
   your branch and the BentoML authors will be notified to review your code
   changes. Once tests are passed and reviewer has signed off, we will merge
   your pull request.


## Optional: Install git hooks to enforce commit format

Run `./dev/install_git_hooks.sh` to install git hooks to automate
commit format enforcement described above.