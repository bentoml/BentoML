import os
import time
import shutil
from simple_di import Provide, inject
from bentoml._internal.configuration.containers import BentoMLContainerClass
from click import confirm
import logging

logger = logging.getLogger(__name__)

@inject
def _get_bundle_store_path(bundle_store_path: str = Provide[BentoMLContainerClass.bundle_store_path]):
    return bundle_store_path
bundle_store_path = _get_bundle_store_path()

def list(name=None):
    """List a set of bundles under a specified name. If a name is not
    specified, all bundles will be listed.

    Args:
        name (str): a name specifying a set of bundles to be deleted.
        skip_confirmation (bool): a flag which specifies whether or not to skip
            deletion confirmation.

    Returns:
        tuple: a list of information about bundles each in the form of a tuple
            of (name, version, creation_time, models).
    """

    names_path = os.path.abspath(bundle_store_path)
    bundles = []
    names = os.scandir(names_path)
    names = names if not name else [n for n in names if n.name == name]
    for n in names:
        name = n.name
        name_path = os.path.join(names_path, name)
        for v in os.scandir(name_path):
            version = v.name
            version_path = os.path.join(name_path, version)
            t = time.ctime(os.path.getctime(version_path))
            models_path = os.path.join(version_path, "models")
            models = sorted([m.name for m in os.scandir(models_path)])
            bundles.append((name,version,str(t),models))
    return bundles

def get(tag):
    """Get a bundle specified by an individual tag.

    Args:
        tag (str): a tag (of the form name:version) specifying one bundle to be
            returned.
    
    Returns:
        tuple: information about a bundle as a tuple in the form of 
            (name, version, creation_time, models).
    """

    name, version = tag.split(":")
    tag_path = os.path.join(bundle_store_path, name, version)
    if not os.path.isdir(tag_path):
        return None
    
    t = time.ctime(os.path.getctime(tag_path))
    models_path = os.path.join(tag_path, "models")
    models = sorted([m.name for m in os.scandir(models_path)])
    return (name,version,str(t),models)

'''
# delete bundles with the following keys:
# tag:  bentoml delete FraudDetector:20210401_EF4C13 --yes
# name: bentoml delete FraudDetector --yes
# labels?
# all?  bentoml delete --all -y
# bentoml.delete(tag, name, labels)
def delete(tag, name, labels):
    pass
'''

def delete(name=None, tag=None, skip_confirmation=None):
    """Delete a set of bundles under a specific name, or an individual tag. If
    neither a tag or name is specified, all bundles will be selected for
    deletion. If skip_confirmation is set to True, then bundles will be deleted
    without confirmation, otherwise, the user will be asked for confirmation of
    deletion.

    Args:
        name (str): a name specifying a set of bundles to be deleted.
        tag (str): a tag (of the form name:version) specifying one bundle to be
            deleted.
        skip_confirmation (bool): a flag which specifies whether or not to skip
            deletion confirmation.
    """

    if tag:
        name, version = tag.split(":")
        tag_path = os.path.join(bundle_store_path, name, version)
        if not os.path.isdir(tag_path):
            return None
        if confirm(f"delete {tag}?"):
            logger.info(f"deleting {tag}")
            shutil.rmtree(tag_path)
        else:
            logger.info(f"skipping {tag}")
    elif name:
        name_path = os.path.join(bundle_store_path, name)
        if not os.path.isdir(name_path):
            return None
        if skip_confirmation:
            logger.info(f"deleting {name}")
            shutil.rmtree(name_path)
        else:
            any_no = False
            for v in os.scandir(name_path):
                version = v.name
                tag = f"{name}:{version}"
                if confirm(f"delete {tag}?"):
                    logger.info(f"deleting {tag}")
                    tag_path = os.path.join(name_path, version)
                    shutil.rmtree(tag_path)
                else:
                    any_no = True
                    logger.info(f"skipping {tag}")
            if not any_no:
                logger.info(f"deleting {name}")
                shutil.rmtree(name_path)
    else:
        if skip_confirmation:
            shutil.rmtree(bundle_store_path)
        else:
            names_path = os.path.abspath(bundle_store_path)
            for n in os.scandir(names_path):
                name = n.name
                name_path = os.path.join(names_path, name)
                for v in os.scandir(name_path):
                    version = v.name
                    version_path = os.path.join(name_path, version)
                    tag = f"{name}:{version}"
                    if confirm(f"delete {tag}?"):
                        logger.info(f"deleting {tag}")
                        shutil.rmtree(version_path)


def push(tag, yatai=None):
    if not yatai:
        # set default yatai server
        pass


def pull(tag, yatai=None):
    if not yatai:
        # set default yatai server  
        pass

''' notes from previous version of file:

#BundleManagar ("~/home/bumdles")
# Add("name", "version") => path

#   returns list of bundles containing name
# bentoml list => show all bundles (from where?)
# bentoml list 'name' => show versions of bundle named 'name' (from where?)
# bentoml.list(name=None) => [Bento]
def list(name=None):
    pass
'''