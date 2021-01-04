set -x

# Set err to 1 if pytest failed.
error=0
trap 'error=1' ERR

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit

python -m pip install -e .

# Run test
python -m pytest -s "$GIT_ROOT"/tests/integration/api_server --docker

test $error = 0 # Return non-zero if pytest failed
