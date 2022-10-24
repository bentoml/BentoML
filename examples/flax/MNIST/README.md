# MNIST classifier

This project demonstrates a simple CNN for MNIST classifier served with BentoML.

### Instruction

Run training scripts:

```bash
bazel run :train -- --num-epochs 2
```

Serve with either gRPC or HTTP:

```bash
bentoml serve-grpc --production --enable-reflection
```

Run the test suite:

```bash
pytest tests
```
