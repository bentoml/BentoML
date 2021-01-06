set -x

# Set err to 1 if pytest failed.
error=0
trap 'error=1' ERR

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit

python -m pip install -e .

# Install minio
pip install minio

# Run s3 tests
python -m pytest -s "$GIT_ROOT"/tests/integration/test_s3.py

test $error = 0 # Return non-zero if pytest failed
