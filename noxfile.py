import os

import nox

os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1", "TOKENIZERS_PARALLELISM": "false"})

TEST_ARGS = [
    "pytest",
    "--cov=bentoml",
    "--cov-report=term-missing",
    "--cov-config=pyproject.toml",
    "-vv",
]

PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11"]


@nox.session(python=PYTHON_VERSIONS, name="unit")
def run_unittest(session: nox.Session):
    session.run("pdm", "sync", "-G", "grpc,io,testing", external=True)
    session.run(*TEST_ARGS, "-n", "auto", "tests/unit")


@nox.session(name="framework-integration")
@nox.parametrize(
    "framework",
    [
        "catboost",
        "diffusers",
        "easyocr",
        "fastai",
        "flax",
        "keras",
        "lightgbm",
        "onnx",
        "picklable_model",
        "pytorch",
        "pytorch_lightning",
        "sklearn",
        "tensorflow",
        "torchscript",
        "xgboost",
        "detectron",
        "transformers",
    ],
)
def run_framework_integration_test(session: nox.Session, framework: str):
    session.run("pdm", "sync", "-G", "testing", external=True)
    session.install(
        "torch",
        "torchaudio",
        "torchvision",
        "-i",
        "https://download.pytorch.org/whl/cpu",
    )
    session.install(
        "catboost",
        "lightgbm",
        "mlflow",
        "fastai",
        "xgboost",
        "scikit-learn",
        "easyocr",
        "datasets",
        # ONNX dependencies
        "onnx",
        "onnxruntime",
        "skl2onnx",
        # tensorflow dependencies
        "tensorflow~=2.13.1",
        # torch-related dependencies
        "lightning",
        # huggingface dependencies
        "transformers",
        "tokenizer",
        "diffusers",
        "flax; platform_system!='Windows'",
        "jax[cpu]; platform_system!='Windows'",
        "jaxlib; platform_system!='Windows'",
        "chex; platform_system!='Windows'",
    )
    session.run(
        *TEST_ARGS,
        "tests/integration/frameworks/test_frameworks.py",
        "--framework",
        framework,
    )


@nox.session(name="e2e-testing", python=PYTHON_VERSIONS)
@nox.parametrize("suite", ["bento_server_http", "bento_server_grpc", "bento_new_sdk"])
def run_e2e_test(session: nox.Session, suite: str):
    session.run("pdm", "sync", "-G", "io,testing", external=True)
    test_folder = os.path.join("tests/e2e", suite)
    requirements = os.path.join(test_folder, "requirements.txt")
    if os.path.exists(requirements):
        session.install("-r", requirements)
    session.run(*TEST_ARGS, test_folder)


@nox.session(name="e2e-monitoring", python=PYTHON_VERSIONS)
def run_e2e_monitoring_test(session: nox.Session):
    session.run("pdm", "sync", "-G", "io,testing,monitor-otlp", external=True)
    test_folder = "examples/monitoring/task_classification"
    session.install("-r", os.path.join(test_folder, "requirements.txt"))
    os.makedirs(os.path.join(test_folder, "monitoring"), exist_ok=True)
    session.run(*TEST_ARGS, test_folder)


@nox.session(name="coverage")
def coverage_report(session: nox.Session):
    session.run("pdm", "sync", "-G", "testing", external=True)
    session.run("coverage", "combine")
    session.run("coverage", "xml")
    session.run("coverage", "html", "--skip-covered", "--skip-empty")
    session.run("python", "tools/generate_coverage.py")
    session.run("python", "tools/write_report.py")
