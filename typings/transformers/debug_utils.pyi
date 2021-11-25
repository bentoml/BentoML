

from .file_utils import ExplicitEnum, is_torch_available

if is_torch_available():
    ...
logger = ...
class DebugUnderflowOverflow:
    """
    This debug class helps detect and understand where the model starts getting very large or very small, and more
    importantly ``nan`` or ``inf`` weight and activation elements.

    There are 2 working modes:

    1. Underflow/overflow detection (default)
    2. Specific batch absolute min/max tracing without detection

    Mode 1: Underflow/overflow detection

    To activate the underflow/overflow detection, initialize the object with the model ::

        debug_overflow = DebugUnderflowOverflow(model)

    then run the training as normal and if ``nan`` or ``inf`` gets detected in at least one of the weight, input or
    output elements this module will throw an exception and will print ``max_frames_to_save`` frames that lead to this
    event, each frame reporting

    1. the fully qualified module name plus the class name whose ``forward`` was run
    2. the absolute min and max value of all elements for each module weights, and the inputs and output

    For example, here is the header and the last few frames in detection report for ``google/mt5-small`` run in fp16 mixed precision ::

        Detected inf/nan during batch_number=0
        Last 21 forward frames:
        abs min  abs max  metadata
        [...]
                          encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
        2.17e-07 4.50e+00 weight
        1.79e-06 4.65e+00 input[0]
        2.68e-06 3.70e+01 output
                          encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
        8.08e-07 2.66e+01 weight
        1.79e-06 4.65e+00 input[0]
        1.27e-04 2.37e+02 output
                          encoder.block.2.layer.1.DenseReluDense.wo Linear
        1.01e-06 6.44e+00 weight
        0.00e+00 9.74e+03 input[0]
        3.18e-04 6.27e+04 output
                          encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
        1.79e-06 4.65e+00 input[0]
        3.18e-04 6.27e+04 output
                          encoder.block.2.layer.1.dropout Dropout
        3.18e-04 6.27e+04 input[0]
        0.00e+00      inf output

    You can see here, that ``T5DenseGatedGeluDense.forward`` resulted in output activations, whose absolute max value
    was around 62.7K, which is very close to fp16's top limit of 64K. In the next frame we have ``Dropout`` which
    renormalizes the weights, after it zeroed some of the elements, which pushes the absolute max value to more than
    64K, and we get an overlow.

    As you can see it's the previous frames that we need to look into when the numbers start going into very large for
    fp16 numbers.

    The tracking is done in a forward hook, which gets invoked immediately after ``forward`` has completed.

    By default the last 21 frames are printed. You can change the default to adjust for your needs. For example ::

        debug_overflow = DebugUnderflowOverflow(model, max_frames_to_save=100)

    To validate that you have set up this debugging feature correctly, and you intend to use it in a training that may
    take hours to complete, first run it with normal tracing enabled for one of a few batches as explained in the next
    section.


    Mode 2. Specific batch absolute min/max tracing without detection

    The second work mode is per-batch tracing with the underflow/overflow detection feature turned off.

    Let's say you want to watch the absolute min and max values for all the ingredients of each ``forward`` call of a
    given batch, and only do that for batches 1 and 3. Then you instantiate this class as ::

        debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1,3])

    And now full batches 1 and 3 will be traced using the same format as explained above. Batches are 0-indexed.

    This is helpful if you know that the program starts misbehaving after a certain batch number, so you can
    fast-forward right to that area.


    Early stopping:

    You can also specify the batch number after which to stop the training, with ::

        debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1,3], abort_after_batch_num=3)

    This feature is mainly useful in the tracing mode, but you can use it for any mode.


    **Performance**:

    As this module measures absolute ``min``/``max`` of each weight of the model on every forward it'll slow the
    training down. Therefore remember to turn it off once the debugging needs have been met.

    Args:
        model (:obj:`nn.Module`):
            The model to debug.
        max_frames_to_save (:obj:`int`, `optional`, defaults to 21):
            How many frames back to record
        trace_batch_nums(:obj:`List[int]`, `optional`, defaults to ``[]``):
            Which batch numbers to trace (turns detection off)
        abort_after_batch_num  (:obj:`int`, `optional`):
            Whether to abort after a certain batch number has finished

    """
    def __init__(self, model, max_frames_to_save=..., trace_batch_nums=..., abort_after_batch_num=...) -> None:
        ...
    
    def save_frame(self, frame=...):
        ...
    
    def expand_frame(self, line):
        ...
    
    def trace_frames(self):
        ...
    
    def reset_saved_frames(self):
        ...
    
    def dump_saved_frames(self):
        ...
    
    def analyse_model(self):
        ...
    
    def analyse_variable(self, var, ctx):
        ...
    
    def batch_start_frame(self):
        ...
    
    def batch_end_frame(self):
        ...
    
    def create_frame(self, module, input, output):
        ...
    
    def register_forward_hook(self):
        ...
    
    def forward_hook(self, module, input, output):
        ...
    


def get_abs_min_max(var, ctx):
    ...

def detect_overflow(var, ctx):
    """
    Report whether the tensor contains any ``nan`` or ``inf`` entries.

    This is useful for detecting overflows/underflows and best to call right after the function that did some math that
    modified the tensor in question.

    This function contains a few other helper features that you can enable and tweak directly if you want to track
    various other things.

    Args:
        var: the tensor variable to check
        ctx: the message to print as a context

    Return:
        :obj:`True` if ``inf`` or ``nan`` was detected, :obj:`False` otherwise
    """
    ...

class DebugOption(ExplicitEnum):
    UNDERFLOW_OVERFLOW = ...
    TPU_METRICS_DEBUG = ...


