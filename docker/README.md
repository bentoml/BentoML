notes:
- GPU supports only for ubuntu, centos

# of images:

- os : 7 &rarr; ubuntu(2004,1804), debian-slim, alpine, centos(8,7), ami
- type: runtime - devel -gpu (for centos & ubuntu)
- python_version: [3.6 3.7 3.8]
- &rarr;  42 images/release per package (model-server, yatai-service)

workflow

  &rarr; parse manifest.yml
  &rarr; get content from partials, and check if the file is correctly format
  &rarr; assemble tags and images given from manifest.yml
  &rarr; Empty our generated directory and generate new Dockerfile

naming convention: ${release_version | devel}-${python_version}-${os}-gpu
  where optional_args can be:
    - cudnn${cudnn_major_version} includes cudart and cudnn
    - cuda${cuda_version} that only contain cudart and cublas
  where os can be:
    - ubuntu${ubuntu_version}
    - alpine${alpine_version}
    - slim
example: 0.13.1-python3.8-ubuntu20.04-gpu
         0.13.1-python3.8-alpine3.14
         0.13.1-python3.7-slim
         0.13.1-python3.6-amazonlinux2

content structure of manifest.yml

    release: dict
      key: str -> type of image {devel/release}
      value: list
        value: defined spec under dist
        -> this will allow us to how to build tags and ensemble our images from partial files
        - for now we are introducing spec system based on supported OS and its corresponding GPU support

    dist: dict
      key: str -> os bentoml going to support
      value: dict -> contains options to parse directly back to release
        add_to_tags: str -> handle tag naming
        write_to_dockerfile: str -> overwrite our generated dockerfile for this given image
        args: list[str] this will be parsed at docker build
        partials: list[str] components needed to build our dockerfile