---
name: Bug Report
about: Create a report to help us improve
title: ''
labels: 'bug'
assignees: ''

---

**Describe the bug**
<!--- A clear and concise description of what the bug is. -->


**To Reproduce**
<!--
Steps to reproduce the issue:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error
-->

**Expected behavior**
<!--- A clear and concise description of what you expected to happen. -->

**Screenshots/Logs**
<!--- 
If applicable, add screenshots, logs or error outputs to help explain your problem.

To give us more information for diagnosing the issue, make sure to enable debug logging:

Add the following lines to your Python code before invoking BentoML:
```python
import bentoml
import logging
bentoml.config().set('core', 'debug', 'true')
bentoml.configure_logging(logging.DEBUG)
```

And use the `--verbose` option when running `bentoml` CLI command, e.g.:
```bash
bentoml get IrisClassifier --verbose
```
-->


**Environment:**
 - OS: [e.g. MacOS 10.14.3]
 - Python/BentoML Version [e.g. Python 3.7.1, BentoML-0.0.5]


**Additional context**
<!-- Add any other context about the problem here. e.g. links to related discussion. -->
