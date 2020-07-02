set -x

# Set err to 1 if pytest failed.
error=0
trap 'error=1' ERR

GIT_ROOT=$(git rev-parse --show-toplevel)
TMP_DIR="$GIT_ROOT/.test_bundle"

cd "$GIT_ROOT" || exit

python -m pip install -e .

## Build docker image
if [ -e "$TMP_DIR" ]; then
	rm -r "$TMP_DIR";
fi
mkdir "$TMP_DIR"
python "$GIT_ROOT"/tests/integration/api_server/example_service.py "$TMP_DIR"

cd "$TMP_DIR"
docker build . -t example_service
docker run -itd --name test_bento_server_mb -p 5000:5000 example_service:latest --workers 1 --enable-microbatch
docker run -itd --name test_bento_server -p 5001:5000 example_service:latest --workers 1
sleep 10

# Run test
python -m pytest --batch-request --host "localhost:5000" "$GIT_ROOT"/tests/integration/api_server
python -m pytest --host "localhost:5001" "$GIT_ROOT"/tests/integration/api_server

docker container stop test_bento_server_mb test_bento_server
docker container rm test_bento_server_mb test_bento_server
test $error = 0 # Return non-zero if pytest failed
