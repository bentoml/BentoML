[<img src="https://raw.githubusercontent.com/bentoml/BentoML/master/docs/source/_static/img/bentoml-readme-header.jpeg" width="600px" margin-left="-5px">](https://github.com/bentoml/BentoML)

# Model Serving Made Easy  [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=BentoML:%20Machine%20Learning%20Model%20Serving%20Made%20Easy%20&url=https://github.com/bentoml/BentoML&via=bentomlai&hashtags=mlops,modelserving,ML,AI,machinelearning,bentoml)

[![pypi_status](https://img.shields.io/pypi/v/bentoml.svg)](https://pypi.org/project/BentoML)
[![downloads](https://pepy.tech/badge/bentoml)](https://pepy.tech/project/bentoml)
[![actions_status](https://github.com/bentoml/bentoml/workflows/BentoML-CI/badge.svg)](https://github.com/bentoml/bentoml/actions)
[![documentation_status](https://readthedocs.org/projects/bentoml/badge/?version=latest)](https://docs.bentoml.org/)
[![join_slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg)

BentoML let you create machine learning powered prediction service in minutes and bridges the gap between data science and DevOps.

## Why BentoML ##

- The easiest way to get your ML models into production.
- High performance model serving, all in Python.
- Package your model once and deploy it anywhere.
- Support all major ML model training [frameworks](https://docs.bentoml.org/en/latest/frameworks.html).

## Getting Started ##

- [Quickstart guide](https://docs.bentoml.org/en/latest/quickstart.html) will show you a simple example of using BentoML in action. In under 10 minutes, you'll be able to serve your ML model over an HTTP API endpoint, and build a docker image that is ready to be deployed in production.
- [Main concepts](https://docs.bentoml.org/en/latest/concepts.html) will give a comprehensive tour of BentoML's components and introduce you to its philosophy. After reading, you will see what drives BentoML's design, and know what `bento` and `runner` stands for.
- Playground notebook gets your hands dirty in a notebook environment, for you to try out all the core features in BentoML.
- [ML Frameworks](https://docs.bentoml.org/en/latest/frameworks.html) lays out best practices and example usages by the ML framework used for training models.
- [Advanced Guides](https://docs.bentoml.org/en/latest/guides/index.html) show cases advanced features in BentoML, including GPU support, inference graph, monitoring, and customizing docker environment etc.

## Community ##

- To report a bug or suggest a feature request, use [GitHub Issues](https://github.com/bentoml/BentoML/issues/new/choose).
- For other discussions, use [Github Discussions](https://github.com/bentoml/BentoML/discussions).
- To receive release announcements, please subscribe to our mailing list or join us on [Slack](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg).

## Contributing ##

There are many ways to contribute to the project:

- If you have any feedback on the project, share it with the community in [Github Discussions](https://github.com/bentoml/BentoML/discussions) of this project.
- Report issues you're facing and "Thumbs up" on issues and feature requests that are relevant to you.
- Investigate bugs and reviewing other developer's pull requests.
- Contributing code or documentation to the project by submitting a Github pull request. See the [development guide](https://github.com/bentoml/BentoML/blob/master/DEVELOPMENT.md).
- See more in the [contributing guide](https://github.com/bentoml/BentoML/blob/master/CONTRIBUTING.md>).

---

### Usage Reporting ###

BentoML by default collects anonymous usage data using [Amplitude](https://amplitude.com/). 
It only collects BentoML library's own actions and parameters, no user or model data will be collected. 
Here is the [code](https://github.com/bentoml/BentoML/blob/master/bentoml/utils/usage_stats.py) that does it.

This helps the BentoML team to understand how the community is using this tool and what to build next. 
You can easily opt-out of usage tracking by running the BentoML commands with the `--do-not-track` option.

    > bentoml [command] --do-not-track

You can also opt-out via setting environment variable `BENTOML_DO_NOT_TRACK=True`

    > export BENTOML_DO_NOT_TRACK=True

### License ###

[Apache License 2.0](https://github.com/bentoml/BentoML/blob/master/LICENSE)

[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fbentoml%2FBentoML.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fbentoml%2FBentoML?ref=badge_large)
