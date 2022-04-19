![bentoml-docker](./templates/docs/bentoml-docker.png)

Make sure to have [buildx](https://docs.docker.com/buildx/working-with-buildx/)
installed.

Install `manager`:
```bash
make install
```

Login to docker, then generate new Dockerfile and README for new bentoml version:
```bash
manager generate --bentoml-version <new_version>
```

Build a given releases:
```bash
manager build --bentoml-version <new_version> --releases runtime --max-worker 5
```
