:; if [ -z 0 ]; then
  goto :WINDOWS
fi

set -x
# Set err to 1 if pytest failed.
error=0
trap 'error=1' ERR

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit

# Run test
PROJECT_PATH="$GIT_ROOT/$1"
BUILD_PATH="$PROJECT_PATH/build"
python "$PROJECT_PATH/model/model.py" "$BUILD_PATH/artifacts"
python "$PROJECT_PATH/service.py" "$BUILD_PATH/artifacts" "$BUILD_PATH/dist"
if [ "$(uname)" == "Darwin" ]; then
	export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
	python -m pytest -s "$PROJECT_PATH" --bento-dist "$BUILD_PATH/dist" --cov=bentoml --cov-config=.coveragerc
else
	python -m pytest -s "$PROJECT_PATH" --bento-dist "$BUILD_PATH/dist" --docker --cov=bentoml --cov-config=.coveragerc
fi
rm -r $BUILD_PATH

test $error = 0 # Return non-zero if pytest failed
exit

:WINDOWS
git rev-parse --show-toplevel > gitroot.txt
set /p GIT_ROOT=<gitroot.txt

cd %GIT_ROOT%

set PROJECT_PATH=%GIT_ROOT%\%1
set BUILD_PATH=%PROJECT_PATH%\build

# Run test
python %PROJECT_PATH%\model\model.py %BUILD_PATH%\artifacts
python %PROJECT_PATH%\service.py %BUILD_PATH%\artifacts %BUILD_PATH%\dist
python -m pytest -s %PROJECT_PATH% --bento-dist %BUILD_PATH%\dist --dev-server --cov=bentoml --cov-config=.coveragerc

rmdir /s /q %BUILD_PATH%
