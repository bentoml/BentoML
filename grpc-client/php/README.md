### Instruction

Make sure to have [`grpc extension`](https://github.com/grpc/grpc/blob/master/src/php/README.md) installed

Generate the stubs:

```bash
./codegen
```

The run the client:

```bash
COMPILED_GRPC_SO=/path/to/grpc.so ./client
```
