

from typing import Callable, Dict, Optional, Tuple

import tensorflow as tf

from .integrations import is_comet_available, is_wandb_available
from .modeling_tf_utils import TFPreTrainedModel
from .trainer_utils import EvalPrediction, PredictionOutput
from .training_args_tf import TFTrainingArguments

if is_wandb_available():
    ...
if is_comet_available():
    ...
logger = ...
class TFTrainer:
    """
    TFTrainer is a simple but feature-complete training and eval loop for TensorFlow, optimized for 🤗 Transformers.

    Args:
        model (:class:`~transformers.TFPreTrainedModel`):
            The model to train, evaluate or use for predictions.
        args (:class:`~transformers.TFTrainingArguments`):
            The arguments to tweak training.
        train_dataset (:class:`~tf.data.Dataset`, `optional`):
            The dataset to use for training. The dataset should yield tuples of ``(features, labels)`` where
            ``features`` is a dict of input features and ``labels`` is the labels. If ``labels`` is a tensor, the loss
            is calculated by the model by calling ``model(features, labels=labels)``. If ``labels`` is a dict, such as
            when using a QuestionAnswering head model with multiple targets, the loss is instead calculated by calling
            ``model(features, **labels)``.
        eval_dataset (:class:`~tf.data.Dataset`, `optional`):
            The dataset to use for evaluation. The dataset should yield tuples of ``(features, labels)`` where
            ``features`` is a dict of input features and ``labels`` is the labels. If ``labels`` is a tensor, the loss
            is calculated by the model by calling ``model(features, labels=labels)``. If ``labels`` is a dict, such as
            when using a QuestionAnswering head model with multiple targets, the loss is instead calculated by calling
            ``model(features, **labels)``.
        compute_metrics (:obj:`Callable[[EvalPrediction], Dict]`, `optional`):
            The function that will be used to compute metrics at evaluation. Must take a
            :class:`~transformers.EvalPrediction` and return a dictionary string to metric values.
        tb_writer (:obj:`tf.summary.SummaryWriter`, `optional`):
            Object to write to TensorBoard.
        optimizers (:obj:`Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.schedules.LearningRateSchedule]`, `optional`):
            A tuple containing the optimizer and the scheduler to use. The optimizer default to an instance of
            :class:`tf.keras.optimizers.Adam` if :obj:`args.weight_decay_rate` is 0 else an instance of
            :class:`~transformers.AdamWeightDecay`. The scheduler will default to an instance of
            :class:`tf.keras.optimizers.schedules.PolynomialDecay` if :obj:`args.num_warmup_steps` is 0 else an
            instance of :class:`~transformers.WarmUp`.
    """
    def __init__(self, model: TFPreTrainedModel, args: TFTrainingArguments, train_dataset: Optional[tf.data.Dataset] = ..., eval_dataset: Optional[tf.data.Dataset] = ..., compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = ..., tb_writer: Optional[tf.summary.SummaryWriter] = ..., optimizers: Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.schedules.LearningRateSchedule] = ...) -> None:
        ...
    
    def get_train_tfdataset(self) -> tf.data.Dataset:
        """
        Returns the training :class:`~tf.data.Dataset`.

        Subclass and override this method if you want to inject some custom behavior.
        """
        ...
    
    def get_eval_tfdataset(self, eval_dataset: Optional[tf.data.Dataset] = ...) -> tf.data.Dataset:
        """
        Returns the evaluation :class:`~tf.data.Dataset`.

        Args:
            eval_dataset (:class:`~tf.data.Dataset`, `optional`):
                If provided, will override `self.eval_dataset`. The dataset should yield tuples of ``(features,
                labels)`` where ``features`` is a dict of input features and ``labels`` is the labels. If ``labels`` is
                a tensor, the loss is calculated by the model by calling ``model(features, labels=labels)``. If
                ``labels`` is a dict, such as when using a QuestionAnswering head model with multiple targets, the loss
                is instead calculated by calling ``model(features, **labels)``.

        Subclass and override this method if you want to inject some custom behavior.
        """
        ...
    
    def get_test_tfdataset(self, test_dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Returns a test :class:`~tf.data.Dataset`.

        Args:
            test_dataset (:class:`~tf.data.Dataset`):
                The dataset to use. The dataset should yield tuples of ``(features, labels)`` where ``features`` is a
                dict of input features and ``labels`` is the labels. If ``labels`` is a tensor, the loss is calculated
                by the model by calling ``model(features, labels=labels)``. If ``labels`` is a dict, such as when using
                a QuestionAnswering head model with multiple targets, the loss is instead calculated by calling
                ``model(features, **labels)``.

        Subclass and override this method if you want to inject some custom behavior.
        """
        ...
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        TFTrainer's init through :obj:`optimizers`, or subclass and override this method.
        """
        ...
    
    def setup_wandb(self):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can subclass and override this method to customize the setup if needed. Find more information `here
        <https://docs.wandb.com/huggingface>`__. You can also override the following environment variables:

        Environment:
            WANDB_PROJECT:
                (Optional): str - "huggingface" by default, set this to a custom string to store results in a different
                project.
            WANDB_DISABLED:
                (Optional): boolean - defaults to false, set to "true" to disable wandb entirely.
        """
        ...
    
    def setup_comet(self):
        """
        Setup the optional Comet.ml integration.

        Environment:
            COMET_MODE:
                (Optional): str - "OFFLINE", "ONLINE", or "DISABLED"
            COMET_PROJECT_NAME:
                (Optional): str - Comet.ml project name for experiments
            COMET_OFFLINE_DIRECTORY:
                (Optional): str - folder to use for saving offline experiments when `COMET_MODE` is "OFFLINE"

        For a number of configurable items in the environment, see `here
        <https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables>`__
        """
        ...
    
    def prediction_loop(self, dataset: tf.data.Dataset, steps: int, num_examples: int, description: str, prediction_loss_only: Optional[bool] = ...) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :func:`~transformers.TFTrainer.evaluate` and
        :func:`~transformers.TFTrainer.predict`.

        Works both with or without labels.
        """
        ...
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
        """
        ...
    
    def evaluate(self, eval_dataset: Optional[tf.data.Dataset] = ...) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        Args:
            eval_dataset (:class:`~tf.data.Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. The dataset should yield tuples of
                ``(features, labels)`` where ``features`` is a dict of input features and ``labels`` is the labels. If
                ``labels`` is a tensor, the loss is calculated by the model by calling ``model(features,
                labels=labels)``. If ``labels`` is a dict, such as when using a QuestionAnswering head model with
                multiple targets, the loss is instead calculated by calling ``model(features, **labels)``.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        ...
    
    def prediction_step(self, features: tf.Tensor, labels: tf.Tensor, nb_instances_in_global_batch: tf.Tensor) -> tf.Tensor:
        """
        Compute the prediction on features and update the loss with labels.

        Subclass and override to inject some custom behavior.
        """
        ...
    
    @tf.function
    def distributed_prediction_steps(self, batch):
        ...
    
    def train(self) -> None:
        """
        Train method to train the model.
        """
        ...
    
    def training_step(self, features, labels, nb_instances_in_global_batch):
        """
        Perform a training step on features and labels.

        Subclass and override to inject some custom behavior.
        """
        ...
    
    def apply_gradients(self, features, labels, nb_instances_in_global_batch):
        ...
    
    @tf.function
    def distributed_training_steps(self, batch):
        ...
    
    def run_model(self, features, labels, training):
        """
        Computes the loss of the given features and labels pair.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            features (:obj:`tf.Tensor`): A batch of input features.
            labels (:obj:`tf.Tensor`): A batch of labels.
            training (:obj:`bool`): Whether or not to run the model in training mode.

        Returns:
            A tuple of two :obj:`tf.Tensor`: The loss and logits.
        """
        ...
    
    def predict(self, test_dataset: tf.data.Dataset) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:class:`~tf.data.Dataset`):
                Dataset to run the predictions on. The dataset should yield tuples of ``(features, labels)`` where
                ``features`` is a dict of input features and ``labels`` is the labels. If ``labels`` is a tensor, the
                loss is calculated by the model by calling ``model(features, labels=labels)``. If ``labels`` is a dict,
                such as when using a QuestionAnswering head model with multiple targets, the loss is instead calculated
                by calling ``model(features, **labels)``

        Returns: `NamedTuple` A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        ...
    
    def save_model(self, output_dir: Optional[str] = ...):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.
        """
        ...
    


