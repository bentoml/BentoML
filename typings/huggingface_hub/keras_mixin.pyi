

from typing import Any, Dict, Optional, Union

from huggingface_hub import ModelHubMixin

logger = ...
def save_pretrained_keras(model, save_directory: str, config: Optional[Dict[str, Any]] = ...): # -> None:
    """Saves a Keras model to save_directory in SavedModel format. Use this if you're using the Functional or Sequential APIs.

    model:
        The Keras model you'd like to save. The model must be compiled and built.
    save_directory (:obj:`str`):
        Specify directory in which you want to save the Keras model.
    config (:obj:`dict`, `optional`):
        Configuration object to be saved alongside the model weights.
    """
    ...

def from_pretrained_keras(*args, **kwargs):
    ...

def push_to_hub_keras(model, repo_path_or_name: Optional[str] = ..., repo_url: Optional[str] = ..., commit_message: Optional[str] = ..., organization: Optional[str] = ..., private: Optional[bool] = ..., api_endpoint: Optional[str] = ..., use_auth_token: Optional[Union[bool, str]] = ..., git_user: Optional[str] = ..., git_email: Optional[str] = ..., config: Optional[dict] = ...): # -> str | Tuple[str, CommandInProgress]:
    """
    Upload model checkpoint or tokenizer files to the 🤗 Model Hub while synchronizing a local clone of the repo in
    :obj:`repo_path_or_name`.

    Parameters:
        model:
            The Keras model you'd like to push to the hub. It model must be compiled and built.
        repo_path_or_name (:obj:`str`, `optional`):
            Can either be a repository name for your model or tokenizer in the Hub or a path to a local folder (in
            which case the repository will have the name of that local folder). If not specified, will default to
            the name given by :obj:`repo_url` and a local directory with that name will be created.
        repo_url (:obj:`str`, `optional`):
            Specify this in case you want to push to an existing repository in the hub. If unspecified, a new
            repository will be created in your namespace (unless you specify an :obj:`organization`) with
            :obj:`repo_name`.
        commit_message (:obj:`str`, `optional`):
            Message to commit while pushing. Will default to :obj:`"add model"`.
        organization (:obj:`str`, `optional`):
            Organization in which you want to push your model or tokenizer (you must be a member of this
            organization).
        private (:obj:`bool`, `optional`):
            Whether or not the repository created should be private (requires a paying subscription).
        api_endpoint (:obj:`str`, `optional`):
            The API endpoint to use when pushing the model to the hub.
        use_auth_token (:obj:`bool` or :obj:`str`, `optional`):
            The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
            generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`). Will default to
            :obj:`True`.
        git_user (``str``, `optional`):
            will override the ``git config user.name`` for committing and pushing files to the hub.
        git_email (``str``, `optional`):
            will override the ``git config user.email`` for committing and pushing files to the hub.
        config (:obj:`dict`, `optional`):
            Configuration object to be saved alongside the model weights.

    Returns:
        The url of the commit of your model in the given repository.
    """
    ...

class KerasModelHubMixin(ModelHubMixin):
    def __init__(self, *args, **kwargs) -> None:
        """
        Mix this class with your keras-model class for ease process of saving & loading from huggingface-hub

        Example::

            >>> from huggingface_hub import KerasModelHubMixin

            >>> class MyModel(tf.keras.Model, KerasModelHubMixin):
            ...    def __init__(self, **kwargs):
            ...        super().__init__()
            ...        self.config = kwargs.pop("config", None)
            ...        self.dummy_inputs = ...
            ...        self.layer = ...
            ...    def call(self, ...)
            ...        return ...

            >>> # Init and compile the model as you normally would
            >>> model = MyModel()
            >>> model.compile(...)
            >>> # Build the graph by training it or passing dummy inputs
            >>> _ = model(model.dummy_inputs)
            >>> # You can save your model like this
            >>> model.save_pretrained("local_model_dir/", push_to_hub=False)
            >>> # Or, you can push to a new public model repo like this
            >>> model.push_to_hub("super-cool-model", git_user="your-hf-username", git_email="you@somesite.com")

            >>> # Downloading weights from hf-hub & model will be initialized from those weights
            >>> model = MyModel.from_pretrained("username/mymodel@main")
        """
        ...
    


