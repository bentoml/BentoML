[<img src="https://raw.githubusercontent.com/bentoml/BentoML/main/docs/source/_static/img/bentoml-readme-header.jpeg" width="600px" margin-left="-5px">](https://github.com/bentoml/BentoML)

<br>

# 🍱 BentoML: Unified Model Serving Framework  [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=BentoML:%20The%20Unified%20Model%20Serving%20Framework%20&url=https://github.com/bentoml&via=bentomlai&hashtags=mlops,bentoml)

[![pypi_status](https://img.shields.io/pypi/v/bentoml.svg)](https://pypi.org/project/BentoML)
[![downloads](https://pepy.tech/badge/bentoml)](https://pepy.tech/project/bentoml)
[![actions_status](https://github.com/bentoml/bentoml/workflows/BentoML-CI/badge.svg)](https://github.com/bentoml/bentoml/actions)
[![documentation_status](https://readthedocs.org/projects/bentoml/badge/?version=latest)](https://docs.bentoml.org/)
[![join_slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](https://join.slack.bentoml.org)

BentoML is an open platform that simplifies ML model deployment and enables you to serve your models at production scale in minutes

👉 [Pop into our Slack community!](https://join.slack.bentoml.org) We're happy to help with any issue you face or even just to meet you and hear what you're working on :)

__The BentoML version 1.0 is around the corner.__ For stable release version 0.13, see
the [0.13-LTS branch](https://github.com/bentoml/BentoML/tree/0.13-LTS). Version 1.0 is 
under active development, you can be of great help by testing out the preview release, 
reporting issues, contribute to the documentation and create sample gallery projects.

## Why BentoML ##

- The easiest way to turn your ML models into production-ready API endpoints.
- High performance model serving, all in Python.
- Standardize model packaging and ML service definition to streamline deployment.
- Support all major machine-learning training [frameworks](https://docs.bentoml.org/en/latest/frameworks/index.html).
- Deploy and operate ML serving workload at scale on Kubernetes via [Yatai](https://github.com/bentoml/yatai).

## Getting Started ##

- [Quickstart guide](https://docs.bentoml.org/en/latest/quickstart.html) will show you a simple example of using BentoML in action. In under 10 minutes, you'll be able to serve your ML model over an HTTP API endpoint, and build a docker image that is ready to be deployed in production.
- [Main concepts](https://docs.bentoml.org/en/latest/concepts/index.html) will give a comprehensive tour of BentoML's components and introduce you to its philosophy. After reading, you will see what drives BentoML's design, and know what `bento` and `runner` stands for.
- [ML Frameworks](https://docs.bentoml.org/en/latest/frameworks/index.html) lays out best practices and example usages by the ML framework used for training models.
- [Advanced Guides](https://docs.bentoml.org/en/latest/guides/index.html) showcases advanced features in BentoML, including GPU support, inference graph, monitoring, and customizing docker environment etc.
- Check out other projects from the [BentoML team](https://github.com/bentoml):
  - [🦄️ Yatai](https://github.com/bentoml/yatai): Run BentoML workflow at scale on Kubernetes
  - [🚀 bentoctl](https://github.com/bentoml/bentoctl): Fast model deployment with BentoML on cloud platforms

## Community ##

- To report a bug or suggest a feature request, use [GitHub Issues](https://github.com/bentoml/BentoML/issues/new/choose).
- For other discussions, use [Github Discussions](https://github.com/bentoml/BentoML/discussions).
- To receive release announcements, please join us on [Slack](https://join.slack.bentoml.org).

## Contributing ##

There are many ways to contribute to the project:

- If you have any feedback on the project, share it with the community in [Github Discussions](https://github.com/bentoml/BentoML/discussions) of this project.
- Report issues you're facing and "Thumbs up" on issues and feature requests that are relevant to you.
- Investigate bugs and reviewing other developer's pull requests.
- Contributing code or documentation to the project by submitting a Github pull request. See the [development guide](https://github.com/bentoml/BentoML/blob/main/DEVELOPMENT.md).
- See more in the [contributing guide](https://github.com/bentoml/BentoML/blob/main/CONTRIBUTING.md).

---

### Usage Reporting ###

By default, BentoML anonymously collect usage data, and here's the [code](./bentoml/_internal/utils/analytics/usage_stats.py) for it. We rely heavily of
community feedback on features and bugs to focus our engineering work on.
As the community grows, we want to ensure we understand how BentoML users
implement and use the library, so that the team can focus on building and
improving BentoML.

Now, we recognize that not everyone is willing to provide and send their usage
data. To opt out, provide `--do-not-track` to any of BentoML CLI commands:
```bash
bentoml serve iris_clf:latest --production --do-not-track
```

Or add the following to your shell `.rc` to opt out tracking entirely:
```bash
# .bashrc.example
export BENTOML_DO_NOT_TRACK=True
```

### License ###

[Apache License 2.0](./LICENSE)

[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fbentoml%2FBentoML.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fbentoml%2FBentoML?ref=badge_large)
