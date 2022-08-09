# PyTorch Mnist with Custom Model Runner

0. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Train and save model:

```bash
python mnist.py --epochs=5
```

2. Start dev server:

```bash
bentoml serve
```

3. Download test data

```bash
wget https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
tar -xf ./mnist_png.tar.gz
```

4. Send test requests

```bash
curl -F 'image=@mnist_png/testing/8/1007.png' http://127.0.0.1:3000/predict
```

5. Load testing

Start production server:
```bash
bentoml serve --production
```

From another terminal:

```bash
pip install locust
locust -H http://0.0.0.0:3000
```
