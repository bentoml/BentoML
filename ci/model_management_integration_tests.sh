set -x

# Set err to 1 if pytest failed.
error=0
trap 'error=1' ERR

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit

python -m pip install -e .

# Run bento management tests
python -m pytest -s "$GIT_ROOT"/tests/integration/model_management

test $error = 0 # Return non-zero if pytest failed
