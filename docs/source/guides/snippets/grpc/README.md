# `Serving with gRPC` snippets

Contains examples for all snippets using for [Serving with gRPC](https://docs.bentoml.org/en/latest/guides/grpc.html)

We will use [`bazel`](bazel.build) to build and run these examples

# Instruction

All clients are built to run with [quickstart image](https://docs.bentoml.org/en/latest/tutorial.html#setup-for-the-tutorial):

```bash
docker run -it --rm -p 8888:8888 -p 3000:3000 -p 3001:3001 bentoml/quickstart:latest serve-grpc --production --enable-reflection
```

To get all available client rules :

```bash
bazel query //... --output label_kind | grep client | sort | column -t
```

The following table contains command to run clients to interact with the quickstart
image:

| Language | Command                           |
| -------- | --------------------------------- |
| Python   | `bazel run //python:client`       |
| C++      | `bazel run //cpp:client`          |
| Go       | `bazel run //go:client`           |
| Java     | `bazel run //java:client`         |
| Node.js  | `pushd js && yarn client && popd` |

# Adding new language support

- Update [gRPC guides](../../grpc.rst)
- Create a new language directory. Add a `client.<ext>` and `BUILD`
- Add new rules to `WORKSPACE`
- `bazel run //:buildifier` for formatting
