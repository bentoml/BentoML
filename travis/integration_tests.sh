set -x

# Set err to 1 if pytest failed.
error=0
trap 'error=1' ERR

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit

python -m pip install -e .

# Run test
#python -m pytest --batch-request --host "localhost:5000" "$GIT_ROOT"/tests/integration/api_server
python -m pytest "$GIT_ROOT"/tests/integration/api_server

test $error = 0 # Return non-zero if pytest failed
