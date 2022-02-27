## BentoML Analytics Tracking

This module will be responsible for providing telemetry on BentoML usage. We rely heavily of
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

Our collection policy:
- Transparency
- Easy to opt out
- We will <b>NOT</b> collect any personal or proprietary data.

## Specification


