# Contributing to BentoML

[BentoML](https://github.com/bentoml/BentoML) is an open and community-driven project. Everyone is welcome to contribute.

The decision-making process and governance structure of BentoML project can be found in the governance document: [BentoML Governance Doc](https://github.com/bentoml/BentoML/blob/main/GOVERNANCE.md).

To follow development updates and discussion, join the #bentoml-contributors channel in [BentoML Slack community](https://join.slack.bentoml.org).


## Ways to contribute

There are many ways to contribute to BentoML.

* Supporting new users by answering questions on the
    [github issues tracker](https://github.com/bentoml/BentoML/issues) and the 
    [#bentoml-users slack channel](https://join.slack.bentoml.org).
 
* Report issues you're facing and "Thumbs up" on issues and feature requests that are 
    relevant to you in BentoML's [issues tracker](https://github.com/bentoml/BentoML/issues).
 
* Investigate bugs and reviewing other developer's pull requests.

* Contributing code or documentation to the project by submitting a Github pull request.

* Create new example projects and contribute it to [bentoml/examples](https://github.com/bentoml/BentoML/tree/main/examples)



## Submitting a bug report or a feature request

We use Github issues to track all bugs and feature requests. Feel free to open an issue
if you have found a bug or wish to see a new feature implemented.

Before submitting a github issue, ensure the bug was not already reported under 
[issues](https://github.com/bentoml/bentoml/issues) or currently being addressed by 
other [pull requests](https://github.com/bentoml/BentoML/pulls).

If you're unable to find an open issue addressing the problem,
[open a new one](https://github.com/bentoml/bentoml/issues/new). Be sure to
include a **title and clear description**, as much relevant information as
possible, and a **code sample** or an **executable test case** demonstrating
the expected behavior that is not occurring.


## Contributing Code

To avoid duplicating work, it is highly recommended to search through the 
[issue tracker](https://github.com/bentoml/bentoml/issues) and 
[pull requests list](https://github.com/bentoml/BentoML/pulls). If in doubt about
duplicated work, or if you want to work on a non-trivial feature, it's recommended to 
first open an issue in the [issue tracker](https://github.com/bentoml/bentoml/issues)
to get some feedbacks from core developers.

One easy way to find an issue to work on is by applying the "help wanted" label in the
issues list: [help wanted issues](https://github.com/bentoml/BentoML/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

For detailed instructions on how to develop BentoML locally and submit a 'pull request',
follow the [development guide](https://github.com/bentoml/BentoML/blob/main/DEVELOPMENT.md).

If you are new to BentoML project and interested in contributing code, take a look at
the [Good first issues list](https://github.com/bentoml/BentoML/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).
Resolving these issues allow you to start contributing to the project without much
prior knoledge and help you get familiar with its codebase.


## Documentation

Improving the documentation is no less important than improving the library. If you find
a typo in the documentation, or have made improvements, do not hesitate to submit a 
GitHub pull request. 

Full documentation can be found under the `docs/source` directory. You can edit the
documentation `.rst` or `.md` files using any text editor. Follow the instructions 
[here](https://github.com/bentoml/BentoML/blob/main/DEVELOPMENT.md#how-to-edit-run-build-documentation-site)
to build documentation site locally, generate HTML output and preview your changes.

## Issue Tracker Tags

Issue type tags:

|     |     |
| --- | --- |
| question | Any questions about the project |
| bug | Something isn't working |
| enhancement | Improving performance, usability, consistency |
| docs | Documentation, tutorials, and example projects |
| new feature | Feature requests or pull request implementing a new feature |
| test | Improving unit test coverage, e2e test, CI or build |

Tags to help new contributors:

|     |     |
| --- | --- |
| help wanted | An issue currently lacks a contributor |
| good first issue | Good for newcomers |

Tags for managing issues:

|     |     |
| --- | --- |
| duplicated | This issue or pull request already exists |
| stale | Automatically applied when an issue went quiet for more than 60 days |
| merge-hold | Requires further discussions before a pull request can be merged |


## Testing and improving test coverage

High quality testing is extremely important for BentoML project. Currently BentoML has
three kind of tests: Unit tests(`tests/`) and integrations (`tests/integration/`) are
running on Travis CI for every pull request.  End-to-end tests(`e2e_tests/`) is manually
executed by the maintainer before every release and for pull requests that are 
introducing major changes.

We expect pull requests that are introducing new features to have at least 90% test 
coverages. Pull requests that are fixing a bug should add a test covering the issue
being fixed if possible.
