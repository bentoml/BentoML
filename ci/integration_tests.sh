set -x

# Set err to 1 if pytest failed.
error=0
trap 'error=1' ERR

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit

# Run test
PROJECT_PATH="$GIT_ROOT/tests/integration/projects/general"
BUILD_PATH="$PROJECT_PATH/build"
python "$PROJECT_PATH/model/model.py" "$BUILD_PATH/artifacts"
python "$PROJECT_PATH/service.py" "$BUILD_PATH/artifacts" "$BUILD_PATH/dist"
if [ "$(uname)" == "Darwin" ]; then
	export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
	python -m pytest -s "$PROJECT_PATH" --bento-dist "$BUILD_PATH/dist"
else
	python -m pytest -s "$PROJECT_PATH" --bento-dist "$BUILD_PATH/dist" --docker
fi
rm -r $BUILD_PATH

# test the non batch service
PROJECT_PATH="$GIT_ROOT/tests/integration/projects/general_non_batch"
BUILD_PATH="$PROJECT_PATH/build"
python "$PROJECT_PATH/model/model.py" "$BUILD_PATH/artifacts"
python "$PROJECT_PATH/service.py" "$BUILD_PATH/artifacts" "$BUILD_PATH/dist"
if [ "$(uname)" == "Darwin" ]; then
	export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
	python -m pytest -s "$PROJECT_PATH" --bento-dist "$BUILD_PATH/dist"
else
	python -m pytest -s "$PROJECT_PATH" --bento-dist "$BUILD_PATH/dist" --docker
fi
rm -r $BUILD_PATH

test $error = 0 # Return non-zero if pytest failed
