


logger = ...
def load_pytorch_checkpoint_in_flax_state_dict(flax_model, pytorch_checkpoint_path, allow_missing_keys=...):
    """Load pytorch checkpoints in a flax model"""
    ...

def convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model):
    ...

def load_flax_checkpoint_in_pytorch_model(model, flax_checkpoint_path):
    """Load flax checkpoints in a PyTorch model"""
    ...

def load_flax_weights_in_pytorch_model(pt_model, flax_state):
    """Load flax checkpoints in a PyTorch model"""
    ...

