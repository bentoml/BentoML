# gRPC client

Contains examples for gRPC clients using for [Serving with gRPC](https://docs.bentoml.org/en/latest/guides/grpc.html)

We will use [`bazel`](bazel.build) to build and run these examples.

If you don't have bazel installed, you can use [./tools/bazel](../tools/bazel) instead.
This script will download a local bazel to `GIT_ROOT` and run bazel.

# Instruction

All client examples are built to run with [quickstart image](https://docs.bentoml.org/en/latest/tutorial.html#setup-for-the-tutorial):

```bash
docker run -it --rm -p 8888:8888 -p 3000:3000 -p 3001:3001 bentoml/quickstart:latest serve-grpc --production --enable-reflection
```

To get all available client rules:

```bash
bazel query //... --output label_kind | grep ":client" | sort | column -t
```

To build all rules for better caching:

```bash
bazel build ...
```

The following table contains command to run clients:

| Language           | Command                                |
| ------------------ | -------------------------------------- |
| [Python](./python) | `bazel run //grpc-client:python`       |
| [C++](./cpp)       | `bazel run //grpc-client:cpp`          |
| [Go](./go)         | `bazel run //grpc-client:go`           |
| [Java](./java)     | `bazel run //grpc-client:java`         |
| [Kotlin](./kotlin) | `bazel run //grpc-client:kotlin`       |
| [Node.js](./node)  | `bazel run //grpc-client:node`         |
| [Swift](./swift)   | `bazel run //grpc-client:swift`        |
| [PHP](./php)       | See [PHP instruction](./php/README.md) |

Note that bazel is first-class support for all gRPC client example. However,
each of the client implementation also support local toolchain. Make sure to modify the given client source for it to work with local tooling.

> For Swift client, make sure to compile gRPC Swift `protoc` beforehand to generate the client stubs.

# Adding new language support

- Update [gRPC guides](../docs/source/guides/grpc.rst)
- Create a new language directory. Add a `client.<ext>` and `BUILD`
- Add new rules to `WORKSPACE`
- `bazel run //:buildifier` for formatting
