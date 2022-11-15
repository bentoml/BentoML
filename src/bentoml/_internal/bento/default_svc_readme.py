readme_template = """# {{ svc.name }}:{{ svc_version }}

[![pypi_status](https://img.shields.io/badge/BentoML-{{ bentoml_version }}-informational)](https://pypi.org/project/BentoML)
[![documentation_status](https://readthedocs.org/projects/bentoml/badge/?version=latest)](https://docs.bentoml.org/)
[![join_slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](https://l.bentoml.com/join-slack-swagger)
[![BentoML GitHub Repo](https://img.shields.io/github/stars/bentoml/bentoml?style=social)](https://github.com/bentoml/BentoML)
[![Twitter Follow](https://img.shields.io/twitter/follow/bentomlai?label=Follow%20BentoML&style=social)](https://twitter.com/bentomlai)

This is a Machine Learning Service created with BentoML.
{% if svc.apis %}
{{ create_inference_api_table(svc) }}


{% endif %}
## Help

* [📖 Documentation](https://docs.bentoml.org/en/latest/): Learn how to use BentoML.
* [💬 Community](https://l.bentoml.com/join-slack-swagger): Join the BentoML Slack community.
* [🐛 GitHub Issues](https://github.com/bentoml/BentoML/issues): Report bugs and feature requests.
* Tip: you can also [customize this README](https://docs.bentoml.org/en/latest/concepts/bento.html#description).
"""
