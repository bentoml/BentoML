# MNIST classifier

This project demonstrates a simple CNN for MNIST classifier served with BentoML.

### Instruction

Run training scripts:

```bash
# run with python3
pip install -r requirements.txt
python3 train.py --num-epochs 2

# run with bazel
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

To run containerize do:

```bash
bentoml containerize mnist_flax --opt platform=linux/amd64
```
