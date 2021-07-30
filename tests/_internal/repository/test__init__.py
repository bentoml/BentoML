from bentoml._internal.repository.__init__ import *

def test_api_fns():
    print(bundle_store_path)

    print("list all:")
    i = 0
    s = time.monotonic()
    for bundle in list():
        if i < 10:
            print(bundle)
        if i > 6000:
            print(bundle)
        i+=1
    e = time.monotonic()
    t = e-s
    print(f"list() w/o tar, {i} tags in {t} seconds @ a rate of {i/t} gets/second")
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

    # tar + non-tar benchmark
    runs = 1000
    s = time.monotonic()
    for i in range(runs):
        get('IrisClassifier:20210618161150_3BFE59')
    e = time.monotonic()
    t = e-s
    print(f"get() w/o tar, {runs} times in {t} seconds @ a rate of {runs/t} gets/second")

    # tag, name -- without yes
    original_tag = "IrisClassifier:20210618161150_3BFE59"
    copy_tag = "IrisClassifierTest:20210618161150_3BFE59"
    original_name, original_version = original_tag.split(':')
    copy_name, copy_version = copy_tag.split(':')
    shutil.copytree(
        os.path.join("/home/alxmke/bentoml/bundles", original_name),
        os.path.join("/home/alxmke/bentoml/bundles", copy_name),
    )
    print(f"deleting by tag, no --yes: {copy_tag}")
    delete(tag=copy_tag)
    print()
    print(f"deleting by name, no --yes: {copy_name}")
    delete(name=copy_name)
    print()

    # tag, name -- with yes
    original_tag = "IrisClassifier:20210618161150_3BFE59"
    copy_tag = "IrisClassifierTest:20210618161150_3BFE59"
    original_name, original_version = original_tag.split(':')
    copy_name, copy_version = copy_tag.split(':')
    shutil.copytree(
        os.path.join("/home/alxmke/bentoml/bundles", original_name),
        os.path.join("/home/alxmke/bentoml/bundles", copy_name),
    )
    delete(tag=copy_tag, yes="y")
    delete(name=copy_name, yes="y")