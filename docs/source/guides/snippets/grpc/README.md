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

The following table contains command to run clients:

| Language           | Command                             |
| ------------------ | ----------------------------------- |
| [Python](./python) | `bazel run //python:client`         |
| [C++](./cpp)       | `bazel run //cpp:client`            |
| [Go](./go)         | `bazel run //go:client`             |
| [Java](./java)     | `bazel run //java:client`           |
| [Kotlin](./kotlin) | `bazel run //kotlin:client`         |
| [Swift](./swift)   | `pushd swift && client && popd`     |
| [Node.js](./node)  | `pushd node && yarn client && popd` |

> For Swift client, make sure to compile gRPC Swift `protoc` beforehand to generate the client stubs.

Note that each of the above client examples are also working standalone if you wish not
to use bazel.

# Adding new language support

- Update [gRPC guides](../../grpc.rst)
- Create a new language directory. Add a `client.<ext>` and `BUILD`
- Add new rules to `WORKSPACE`
- `bazel run //:buildifier` for formatting

### TODO:

- Write ruleset for compiling Swift
- Write ruleset for running grpc-node
