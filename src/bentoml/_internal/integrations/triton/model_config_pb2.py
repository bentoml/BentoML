from google.protobuf import __version__

if __version__.startswith("4"):
    from ._generated_pb4.model_config_pb2 import *
else:
    from ._generated_pb3.model_config_pb2 import *
