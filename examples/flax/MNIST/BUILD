load("@bazel_skylib//rules:write_file.bzl", "write_file")

write_file(
    name = "_train_sh",
    out = "_train.sh",
    content = [
        "#!/usr/bin/env bash\n",
        "cd $BUILD_WORKING_DIRECTORY\n",
        "python -m pip install -r requirements.txt\n",
        "python train.py $@",
    ],
)

sh_binary(
    name = "train",
    srcs = ["_train.sh"],
    data = ["train.py"],
)
