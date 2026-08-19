# Runtime environment spec — deep reference

Two equivalent ways to define a Bento's runtime environment. Prefer the
in-code `bentoml.images.Image` API for new projects; `bentofile.yaml` remains
fully supported.

## Option A: `bentoml.images.Image` (in `service.py`)

```python
import bentoml

my_image = (
    bentoml.images.Image(python_version="3.11")
    .python_packages("torch", "transformers")
)

@bentoml.service(image=my_image)
class MyService:
    ...
```

### Constructor arguments

| Argument | Default | Meaning |
|---|---|---|
| `python_version` | current build env's Python | e.g. `"3.11"` |
| `distro` | `"debian"` | base image family |
| `base_image` | `""` | custom Docker base image; overrides `python_version`/`distro` |
| `lock_python_packages` | `True` | lock all versions (incl. transitive deps) at build time; set `False` when you pin versions yourself |

### Chainable methods (order-sensitive — commands run in the order declared)

- `.python_packages("numpy>=1.20", "git+https://github.com/user/repo.git@branch", "./wheels/pkg-0.1.0-py3-none-any.whl")`
  — PyPI specs, Git URLs, or local wheels. The `wheels/` directory is included
  in the Bento automatically.
- `.requirements_file("./requirements.txt")` — use an existing requirements file.
- `.pyproject_toml("pyproject.toml")` — take dependencies from a pyproject.
- `.system_packages("curl", "git")` — distro package manager installs.
- `.run('command')` — arbitrary build-time shell command. Placement matters:
  `.run(...)` before `.python_packages(...)` executes before pip install, after
  it executes after.
- `.run_script("scripts/setup.sh")` — run a script file (needs a shebang).
- `.build_include("data/", "config/settings.yaml")` — extra paths copied into
  the build context before python packages are installed.

Note: in a multi-service Bento all services share one runtime environment; a
per-service image is not supported.

## Option B: `bentofile.yaml`

Placed next to `service.py`. Only `service:` is required.

```yaml
service: "service:MyService"      # <module>:<class name>
name: my-bento                    # optional; default = service class name in snake_case (MyService -> my_service)
description: "..."
labels:
  owner: my-team
include:                          # files to package, relative to build ctx (default: everything)
  - "*.py"
  - "config/"
exclude:                          # applied after include
  - "tests/"
python:
  packages:
    - torch
    - transformers
  requirements_txt: "./requirements.txt"   # alternative to packages
  lock_packages: true
docker:
  distro: debian                  # debian|alpine|ubi8|amazonlinux
  python_version: "3.11"
  system_packages:
    - curl
  base_image: ""                  # custom base image (advanced)
  cuda_version: "12.1"            # GPU images; incompatible with conda
# (a miniconda-based image variant is selected automatically when top-level
#  `conda:` options are present; there is no separate distro value for it)
envs:                             # env vars baked as ENV into the image
  - name: HF_HUB_OFFLINE
    value: "1"
  - name: HF_TOKEN                # name only => value supplied at runtime
models:                           # models from the local model store to package
  - my_model:latest
args: {}                          # template args for parametrized builds
```

Unknown top-level keys are rejected (`__forbid_extra_keys__`), so typos fail the
build with a clear message.

Build with a non-default file name/location: `bentoml build -f path/to/bentofile.yaml`.
Other useful flags: `--name`, `--version` (default auto-generated), `--label KEY=VALUE`.

## `.bentoignore`

Gitignore-style, evaluated per directory recursively (the file can exist at any
level of the build context; patterns are relative to the directory containing
it). Note that `.git/`, `.venv/`, `venv/`, `__pycache__/`, and `.DS_Store` are
**always excluded automatically** — you don't need to list them. Use
`.bentoignore` for large data/checkpoint dirs and other artifacts:

```
data/
checkpoints/
*.ipynb
```

`include`/`exclude` in `bentofile.yaml` and `.bentoignore` compose: a file is
packaged only if it matches `include`, does not match `exclude`, and is not
ignored.

## Environment variables

- `envs` entries **with a value** become `ARG`/`ENV` lines in the generated
  Dockerfile — they are baked into image layers and visible to anyone with the
  image. Never bake secrets.
- `envs` entries **without a value** (name only) document what must be supplied
  at runtime (`docker run -e`, Kubernetes Secret/env).
- The same list can be given in code: `@bentoml.service(envs=[{"name": "HF_TOKEN"}])`.

## Models: baked into the image vs downloaded at runtime

There are three patterns; know which one the user's service uses because it
changes both containerize behavior and what the Kubernetes deployment needs.

### 1. Declared model references (recommended) — baked into the image

```python
import bentoml
from bentoml.models import BentoModel, HuggingFaceModel

@bentoml.service()
class MyService:
    # MUST be class variables, not created inside __init__ —
    # class-level declaration registers them as Bento dependencies.
    model_path = HuggingFaceModel("google-bert/bert-base-uncased")  # returns a path
    sk_ref = BentoModel("iris_sklearn:latest")                       # local model store

    def __init__(self):
        from transformers import AutoModelForSequenceClassification
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
```

During `bentoml containerize`, BentoML resolves **all** declared models —
downloading Hugging Face snapshots and copying model-store models — into the
image build context, so the final image contains the weights (HF models under
`$BENTO_PATH/hf-models`, wired up via the baked `BENTOML_HF_CACHE_DIR` env var;
model-store models under `$BENTO_PATH/models`). Consequences:

- No network or token is needed **at runtime** to fetch weights.
- The image can be multiple GB. Registry and cluster must tolerate the size.
- **Gated/private HF models**: export the token in the shell that runs the
  build/containerize commands: `export HF_TOKEN=hf_...` (huggingface_hub reads
  it). The token is used at build time only and is NOT stored in the image.
- `HuggingFaceModel(model_id, revision="main", include=[...], exclude=[...])`
  can trim what gets downloaded (e.g. exclude `*.bin` when safetensors exist).

### 2. Ad-hoc runtime downloads — NOT baked

If the service calls e.g. `from_pretrained("some/model")` in `__init__` without
a class-level `HuggingFaceModel` reference, nothing is packaged; the container
downloads the model **every time it starts**. This works, but:

- The pod needs outbound network access and any auth token at runtime
  (`docker run -e HF_TOKEN=...` locally; a Kubernetes Secret in the cluster —
  tell the deploy skill).
- Startup is slow and unreliable; prefer converting to pattern 1.

### 3. Models from the local model store (`bentofile.yaml` `models:` list)

Same baking behavior as `BentoModel` above; entries reference tags in the local
store (`bentoml models list`).
