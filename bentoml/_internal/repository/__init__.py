from genericpath import isdir
import os
import time
import shutil
from bentoml._internal.configuration.containers import BentoMLContainerClass  # need to change to explicit path after in-file testing

# home alxmke -> bentoml home; bentoml/_internal/configuration/containers
from click import confirm, echo

from simple_di import Provide
bentoml_home = BentoMLContainerClass.bentoml_home._provide()

#root_bundle_store: str = Provide[
#    BentoMLContainerClass.bentoml_home
#]+"/bundles/"

def list(name=None):
    names_path = os.path.abspath(bentoml_home)
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
    name, version = tag.split(":")
    tag_path = os.path.join(bentoml_home, name, version)
    if not os.path.isdir(tag_path):
        return None
    
    t = time.ctime(os.path.getctime(tag_path))
    models_path = os.path.join(tag_path, "models")
    models = sorted([m.name for m in os.scandir(models_path)])
    return (name,version,str(t),models)

def delete(name=None, tag=None, yes=None):
    if tag:
        name, version = tag.split(":")
        tag_path = os.path.join(bentoml_home, name, version)
        if not os.path.isdir(tag_path):
            return None
        if confirm(f"delete {tag}?"):
            echo(f"deleting {tag}")
            shutil.rmtree(tag_path)
        else:
            echo(f"skipping {tag}")
    elif name:
        name_path = os.path.join(bentoml_home, name)
        if not os.path.isdir(name_path):
            return None
        if yes:
            echo(f"deleting {name}")
            shutil.rmtree(name_path)
        else:
            any_no = False
            for v in os.scandir(name_path):
                version = v.name
                tag = f"{name}:{version}"
                if confirm("delete {tag}?"):
                    echo("deleting {tag}")
                    tag_path = os.path.join(name_path, version)
                    shutil.rmtree(tag_path)
                else:
                    any_no = True
                    echo("skipping {tag}")
            if not any_no:
                echo(f"deleting {name}")
                shutil.rmtree(name_path)

def push(tag, yatai=None):
    if not yatai:
        # set default yatai server
        pass


def pull(tag, yatai=None):
    if not yatai:
        # set default yatai server  
        pass

''' notes from previous version of file:

# name:version = tag

#BundleManagar ("~/home/bumdles")
# Add("name", "version") => path

# See all the bundles I built during development
#   returns list of bundles containing name
# bentoml list => show all bundles (from where?)
# bentoml list 'name' => show versions of bundle named 'name' (from where?)
# bentoml.list(name=None) => [Bento]
def list(name=None):
    pass
    #bentoml/yatai/yatai_service_impl.py -- ListBento

# Show metadata for bundle with tag 'tag'
# - (which part of 'FraudDetector:20210401_EF4C13' contains tag?)
#   returns metadata (form of what?)
# bentoml get FraudDetector:20210401_EF4C13 => show bundle metadata
# betnoml.get(tag) => Bento
def get(tag):
    pass

# delete bundles with the following keys:
# tag:  bentoml delete FraudDetector:20210401_EF4C13 --yes
# name: bentoml delete FraudDetector --yes
# labels?
# all?  bentoml delete --all -y
# betnoml.delete(tag, name, labels)
def delete(tag, name, labels):
    pass

# bentoml push FraudDetector:20210401_EF4C13 --yatai=URL
# betnoml.push(tag, yatai=..)
def delete(tag, yatai):
    pass

# bentoml pull FraudDetector:20210401_EF4C13
# betnoml.pull(tag, yatai=..)
def pull(tag, yatai):
    pass

# Containerize to docker image
# bentoml.containerize(tag | bundle_path, image_tag="..", ..)
def containerize(tag_or_bundle_path, image_tag):
    pass

# bentoml.serve(tag | bundle_path, mode=development, ..)
def serve(tag_or_bundle_path, mode="development"):
    pass
'''