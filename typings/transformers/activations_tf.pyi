import tensorflow as tf
from packaging import version

def mish(x): ...
def gelu_fast(x): ...

if version.parse(tf.version.VERSION) >= version.parse("2.4"):
    def approximate_gelu_wrap(x): ...
    gelu = ...
    gelu_new = ...
else:
    gelu = ...
    gelu_new = ...
ACT2FN = ...
