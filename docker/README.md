![bentoml-docker](./templates/docs/bentoml-docker.png)

Make sure to have [buildx](https://docs.docker.com/buildx/working-with-buildx/)
installed.

Install `manager`:
```bash
make install
```

Generate new Dockerfile and README for new bentoml version:
```bash
manager generate --bentoml-version <new_version>
```

Build 
