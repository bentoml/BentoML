import os
import yaml
import time
import tarfile

def list(name=None):
    # TODO: change bundles path to general path
    names_path = os.path.abspath("/home/alxmke/bentoml/bundles")
    timev1 = True
    timev2 = False
    bundles = []
    names = os.scandir(names_path)
    names = names if not name else [n for n in names if n.name == name]
    for n in names:
        name = n.name
        name_path = os.path.join(names_path, name)
        for v in os.scandir(name_path):
            version = v.name
            t = None
            models = None
            if timev1:  # ~60k tags/second
                version_path = os.path.join(name_path, version)
                t = time.ctime(os.path.getctime(version_path))
                models_path = os.path.join(name_path, version, "models")
                models = sorted([m.name for m in os.scandir(models_path)])
            elif timev2:  # ~460 tags/second
                bento_yaml_path = os.path.join(name_path, version, "bentoml.yml")
                bento_yaml = None
                with open(bento_yaml_path, 'r') as f:
                    bento_yaml = yaml.load(f, Loader=yaml.FullLoader)
                t = bento_yaml["metadata"]["created_at"]
                # TODO: when artifacts->models in yaml, change here 
                models = [model["name"] for model in bento_yaml["artifacts"]]
            # times are mildly different between versions, will probably pick one
            bundles.append((name,version,str(t),models))
    return bundles

# TODO: look into scandir stat

def get(tag):
    # what kind of error to throw if tag malformed?
    # and if it doesn't exist?
    name, version = tag.split(":")
    # TODO: change bundles path to general path
    tag_path = os.path.join("/home/alxmke/bentoml/bundles", name, version)
    if not os.path.isdir(tag_path):
        return None
    
    timev1 = False
    timev2 = True
    t = None
    models = None
    if timev1:  # ~60k tags/second
        t = time.ctime(os.path.getctime(tag_path))
        models_path = os.path.join(tag_path, "models")
        models = sorted([m.name for m in os.scandir(models_path)])
    elif timev2:  # ~460 tags/second
        bento_yaml_path = os.path.join(tag_path, "bentoml.yml")
        bento_yaml = None
        with open(bento_yaml_path, 'r') as f:
            bento_yaml = yaml.load(f, Loader=yaml.FullLoader)
        t = bento_yaml["metadata"]["created_at"]
        models = [model["name"] for model in bento_yaml["artifacts"]]
    return (name,version,str(t),models)

def get_tar(tag):
    # what kind of error to throw if tag malformed?
    # and if it doesn't exist?
    name, version = tag.split(":")
    # TODO: change bundles path to general path
    tag_path = os.path.join("/home/alxmke/bentoml/bundles-tar", name, version+".bento")
    if not os.path.isfile(tag_path):
        return None
    
    timev1 = False
    timev2 = True
    tf = tarfile.open(tag_path, mode='r')
    bento_yaml = yaml.load(
        tf.extractfile(os.path.join(version, "bentoml.yml")),
        Loader=yaml.FullLoader,
    )
    t = bento_yaml["metadata"]["created_at"]
    models = [model["name"] for model in bento_yaml["artifacts"]]
    tf.close()

    return (name,version,str(t),models)
    
# BentoProtoBuffMessage
# yatai client proto bentoml.yatai.proto.

# bentoml delete [--yes] <-
def delete(name=None, tag=None):
    if tag:
        path = bundles(tag=tag).get("path")
    if name:
        pass

def push(tag, yatai=None):
    if not yatai:
        # set default yatai server
        pass


def pull(tag, yatai=None):
    if not yatai:
        # set default yatai server  
        pass

# check benchmarks and proposals from:
# https://stackoverflow.com/questions/3964681/find-all-files-in-a-directory-with-extension-txt-in-python
# for:
# returns all bundles with tag, created time, path?
# walk the bentoml/bundles -- ~/bentoml/bundles/* for now, but later
# should be a relative path rather than home directory 
def bundles(name=None):
    dir = "/home"
    npath = os.path.abspath(dir + "/bentoml/repository")

    print("name", "\t\t\tversion")
    for n in os.scandir(npath):
        name = n.name
        vpath = os.path.join(npath, name)
        for v in os.scandir(vpath):
            version = v.name
            print(name, version)


i = 0
print("list all:")
for bundle in list():
    if i < 10:
        print(bundle)
    if i > 6000:
        print(bundle)
    i+=1
print()

print("list all w/ name IrisClassifier642:")
i = 0
for bundle in list("IrisClassifier642"):
    print(bundle)
    i+=1
print()

print("get (IrisClassifier642:20210618161150_3BFE59):")
print("bundle:", get('IrisClassifier642:20210618161150_3BFE59'))
print()

print("get (THIS_NAME:DOES_NOT_EXIST):")
print("bundle:", get('THIS_NAME:DOES_NOT_EXIST'))
print()

print("get_tar (IrisClassifier:20210618161150_3BFE59):")
print("bundle:", get_tar('IrisClassifier:20210618161150_3BFE59'))
print()

print("get_tar (THIS_NAME:DOES_NOT_EXIST):")
print("bundle:", get_tar('THIS_NAME:DOES_NOT_EXIST'))
print()

# tar + non-tar benchmark
runs = 1000
s = time.monotonic()
for i in range(runs):
    get('IrisClassifier:20210618161150_3BFE59')
e = time.monotonic()
t = e-s
print(f"get() w/o tar, {runs} times in {t} seconds @ a rate of {runs/t} gets/second")

s = time.monotonic()
for i in range(runs):
    get_tar('IrisClassifier:20210618161150_3BFE59')
e = time.monotonic()
t = e-s
print(f"get() w/ tar, {runs} times in {t} seconds @ a rate of {runs/t} gets/second")


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

'''
import os
import yaml

names_path = os.path.abspath("/home/alxmke/bentoml/bundles")
i = 0
import time
s = time.monotonic()
print("name", "\t\t\tversion", "\t\ttime", "\t\t\t\tartifacts")


timev1 = False
timev2 = True

for n in os.scandir(names_path):
	name = n.name
	name_path = os.path.join(names_path, name)
	for v in os.scandir(name_path):
		version = v.name
		t = None
		artifacts = None
		if timev1:  # ~200k tags/second
			version_path = os.path.join(name_path, version)
			t = time.ctime(os.path.getctime(version_path))
		elif timev2:  # ~460 tags/second
			bento_yaml_path = os.path.join(name_path, version, "bentoml.yml")
			bento_yaml = None
			with open(bento_yaml_path, 'r') as f:
				bento_yaml = yaml.load(f, Loader=yaml.FullLoader)
			t = bento_yaml["metadata"]["created_at"]
			artifacts = bento_yaml["artifacts"]
		if i < 20:
			print(name, version, t, artifacts, sep="\t")
		elif i == 20:
			print("...")
		elif i > 979 and i < 1000:
			print(name, version, t, artifacts, sep="\t")
		i+=1
e = time.monotonic()
t = e-s
print(f"\n\ndiscovered and printed {i} tags in {t} seconds @ a rate of {i/t} tags/second")
'''