import os
import sys

from ruamel.yaml import YAML

from bentoml.service import BentoService
from bentoml.version import __version__ as BENTOML_VERSION


class _LoadedBentoServiceWrapper(BentoService):

    def __init__(self, model_service_class, path, config):
        super(_LoadedBentoServiceWrapper, self).__init__()
        self.path = path
        self.config = config
        self.model_service = model_service_class()
        self.loaded = False
        self._wrap_api_funcs()

    def load(self, path=None):
        if path is not None:
            # TODO: warn user path is ignored when using pip installed bentoML model
            pass
        self.model_service.load(self.path)
        self.loaded = True

    @property
    def apis(self):
        return self.model_service._apis

    @property
    def name(self):
        return self.config['model_name']

    @property
    def version(self):
        return self.config['model_version']

    def _wrap_api_funcs(self):
        """
        Add target ModelService's API methods to the returned wrapper class
        """
        for api in self.apis:
            setattr(self, api.name,
                    self.model_service.__getattribute__(api.name).__get__(self.model_service))


def load_bentoml_config(path):
    try:
        with open(os.path.join(path, 'bentoml.yml'), 'r') as f:
            bentml_yml_content = f.read()
    except FileNotFoundError:
        raise ValueError("Bentoml can't locate model config file 'bentoml.yml' "
                         " in the give path: {}".format(path))

    yaml = YAML()
    # load bentoml.yml as config
    bentoml_config = yaml.load(bentml_yml_content)

    if bentoml_config['bentoml_version'] != BENTOML_VERSION:
        # TODO: warn about bentoml version mismatch
        pass

    # TODO: also check python version here

    return bentoml_config


def load(path, lazy_load=False):
    """
    Load a BentoService or BentoModel from saved archive in given path

    :param path: A BentoArchive path generated from BentoService.save call
    :return: BentoService
    """
    config = load_bentoml_config(path)

    # Load target module containing BentoService class from given path
    module_file_path = os.path.join(path, config['model_name'], config['module_file'])

    if sys.version_info >= (3, 5):
        import importlib.util
        spec = importlib.util.spec_from_file_location(config['module_name'], module_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    elif sys.version_info >= (3, 3):
        from importlib.machinery import SourceFileLoader
        # pylint:disable=deprecated-method
        module = SourceFileLoader(config['module_name'],
                                  module_file_path).load_module(config['module_name'])
        # pylint:enable=deprecated-method
    else:
        import imp
        module = imp.load_source(config['module_name'], module_file_path)

    model_service_class = module.__getattribute__(config['model_name'])
    loaded_model = _LoadedBentoServiceWrapper(model_service_class, path, config)

    if not lazy_load:
        loaded_model.load()

    return loaded_model
