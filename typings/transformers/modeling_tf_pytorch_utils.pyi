


logger = ...
def convert_tf_weight_name_to_pt_weight_name(tf_name, start_prefix_to_remove=...):
    """
    Convert a TF 2.0 model variable name in a pytorch model weight name.

    Conventions for TF2.0 scopes -> PyTorch attribute names conversions:

        - '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)
        - '_._' is replaced by a new level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)

    return tuple with:

        - pytorch model weight name
        - transpose: boolean indicating whether TF2.0 and PyTorch weights matrices are transposed with regards to each
          other
    """
    ...

def load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path, tf_inputs=..., allow_missing_keys=...):
    """Load pytorch checkpoints in a TF 2.0 model"""
    ...

def load_pytorch_model_in_tf2_model(tf_model, pt_model, tf_inputs=..., allow_missing_keys=...):
    """Load pytorch checkpoints in a TF 2.0 model"""
    ...

def load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict, tf_inputs=..., allow_missing_keys=...):
    """Load pytorch state_dict in a TF 2.0 model."""
    ...

def load_tf2_checkpoint_in_pytorch_model(pt_model, tf_checkpoint_path, tf_inputs=..., allow_missing_keys=...):
    """
    Load TF 2.0 HDF5 checkpoint in a PyTorch model We use HDF5 to easily do transfer learning (see
    https://github.com/tensorflow/tensorflow/blob/ee16fcac960ae660e0e4496658a366e2f745e1f0/tensorflow/python/keras/engine/network.py#L1352-L1357).
    """
    ...

def load_tf2_model_in_pytorch_model(pt_model, tf_model, allow_missing_keys=...):
    """Load TF 2.0 model in a pytorch model"""
    ...

def load_tf2_weights_in_pytorch_model(pt_model, tf_weights, allow_missing_keys=...):
    """Load TF2.0 symbolic weights in a PyTorch model"""
    ...

