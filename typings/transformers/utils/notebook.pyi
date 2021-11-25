

from typing import Optional

from ..trainer_callback import TrainerCallback

def format_time(t): # -> str:
    "Format `t` (in seconds) to (h):mm:ss"
    ...

def html_progress_bar(value, total, prefix, label, width=...): # -> str:
    ...

def text_to_html_table(items): # -> str:
    "Put the texts in `items` in an HTML table."
    ...

class NotebookProgressBar:
    """
    A progress par for display in a notebook.

    Class attributes (overridden by derived classes)

        - **warmup** (:obj:`int`) -- The number of iterations to do at the beginning while ignoring
          :obj:`update_every`.
        - **update_every** (:obj:`float`) -- Since calling the time takes some time, we only do it every presumed
          :obj:`update_every` seconds. The progress bar uses the average time passed up until now to guess the next
          value for which it will call the update.

    Args:
        total (:obj:`int`):
            The total number of iterations to reach.
        prefix (:obj:`str`, `optional`):
            A prefix to add before the progress bar.
        leave (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to leave the progress bar once it's completed. You can always call the
            :meth:`~transformers.utils.notebook.NotebookProgressBar.close` method to make the bar disappear.
        parent (:class:`~transformers.notebook.NotebookTrainingTracker`, `optional`):
            A parent object (like :class:`~transformers.utils.notebook.NotebookTrainingTracker`) that spawns progress
            bars and handle their display. If set, the object passed must have a :obj:`display()` method.
        width (:obj:`int`, `optional`, defaults to 300):
            The width (in pixels) that the bar will take.

    Example::

        import time

        pbar = NotebookProgressBar(100)
        for val in range(100):
            pbar.update(val)
            time.sleep(0.07)
        pbar.update(100)
    """
    warmup = ...
    update_every = ...
    def __init__(self, total: int, prefix: Optional[str] = ..., leave: bool = ..., parent: Optional[NotebookTrainingTracker] = ..., width: int = ...) -> None:
        ...
    
    def update(self, value: int, force_update: bool = ..., comment: str = ...): # -> None:
        """
        The main method to update the progress bar to :obj:`value`.

        Args:

            value (:obj:`int`):
                The value to use. Must be between 0 and :obj:`total`.
            force_update (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force and update of the internal state and display (by default, the bar will wait for
                :obj:`value` to reach the value it predicted corresponds to a time of more than the :obj:`update_every`
                attribute since the last update to avoid adding boilerplate).
            comment (:obj:`str`, `optional`):
                A comment to add on the left of the progress bar.
        """
        ...
    
    def update_bar(self, value, comment=...): # -> None:
        ...
    
    def display(self): # -> None:
        ...
    
    def close(self): # -> None:
        "Closes the progress bar."
        ...
    


class NotebookTrainingTracker(NotebookProgressBar):
    """
    An object tracking the updates of an ongoing training with progress bars and a nice table reporting metrics.

    Args:

        num_steps (:obj:`int`): The number of steps during training.
        column_names (:obj:`List[str]`, `optional`):
            The list of column names for the metrics table (will be inferred from the first call to
            :meth:`~transformers.utils.notebook.NotebookTrainingTracker.write_line` if not set).
    """
    def __init__(self, num_steps, column_names=...) -> None:
        ...
    
    def display(self): # -> None:
        ...
    
    def write_line(self, values): # -> None:
        """
        Write the values in the inner table.

        Args:
            values (:obj:`Dict[str, float]`): The values to display.
        """
        ...
    
    def add_child(self, total, prefix=..., width=...): # -> NotebookProgressBar:
        """
        Add a child progress bar displayed under the table of metrics. The child progress bar is returned (so it can be
        easily updated).

        Args:
            total (:obj:`int`): The number of iterations for the child progress bar.
            prefix (:obj:`str`, `optional`): A prefix to write on the left of the progress bar.
            width (:obj:`int`, `optional`, defaults to 300): The width (in pixels) of the progress bar.
        """
        ...
    
    def remove_child(self): # -> None:
        """
        Closes the child progress bar.
        """
        ...
    


class NotebookProgressCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that displays the progress of training or evaluation, optimized for
    Jupyter Notebooks or Google colab.
    """
    def __init__(self) -> None:
        ...
    
    def on_train_begin(self, args, state, control, **kwargs): # -> None:
        ...
    
    def on_step_end(self, args, state, control, **kwargs): # -> None:
        ...
    
    def on_prediction_step(self, args, state, control, eval_dataloader=..., **kwargs): # -> None:
        ...
    
    def on_log(self, args, state, control, logs=..., **kwargs): # -> None:
        ...
    
    def on_evaluate(self, args, state, control, metrics=..., **kwargs):
        ...
    
    def on_train_end(self, args, state, control, **kwargs): # -> None:
        ...
    


