import string
import typing as t
import logging
import operator
import functools
from typing import TYPE_CHECKING

import yaml
import click
from cerberus import Validator as CerberusValidator
from manager._utils import SUPPORTED_OS_RELEASES

if TYPE_CHECKING:

    from manager._types import GenericDict
    from manager._types import ValidateType
    from manager._types import CerberusValidator  # pylint: disable


logger = logging.getLogger(__name__)

SPEC_SCHEMA = r"""
---
specs:
  required: true
  type: dict
  schema:
    releases:
      required: true
      type: dict
      schema:
        templates_dir:
          type: string
          nullable: False
        base_image:
          type: string
          nullable: False
        add_to_tags:
          type: string
          nullable: False
          required: true
        multistage_image:
          type: boolean
        header:
          type: string
        envars:
          type: list
          forbidden: ['HOME']
          default: []
          schema:
            check_with: args_format
            type: string
        cuda_prefix_url:
          type: string
          dependencies: cuda
        cuda_requires:
          type: string
          dependencies: cuda
        cuda:
          type: dict
          matched: 'specs.dependencies.cuda'
    tag:
      required: True
      type: dict
      schema:
        fmt:
          type: string
          check_with: keys_format
          regex: '({(.*?)})'
        release_type:
          type: string
          nullable: True
        python_version:
          type: string
          nullable: True
        suffixes:
          type: string
          nullable: True
    dependencies:
      type: dict
      keysrules:
        type: string
      valuesrules:
        type: dict
        keysrules:
          type: string
        valuesrules:
          type: string
          nullable: true
    repository:
      type: dict
      schema:
        pwd:
          env_vars: true
          type: string
        user:
          dependencies: pwd
          env_vars: true
          type: string
        urls:
          keysrules:
            type: string
          valuesrules:
            type: string
            regex: '((([A-Za-z]{3,9}:(?:\/\/)?)(?:[-;:&=\+\$,\w]+@)?[A-Za-z0-9.-]+|(?:www.|[-;:&=\+\$,\w]+@)[A-Za-z0-9.-]+)((?:\/[\+~%\/.\w\-_]*)?\??(?:[-\+=&;%@.\w_]*)#?(?:[\w]*))?)'
        registry:
          keysrules:
            type: string
          type: dict
          valuesrules:
            type: string

repository:
  type: dict
  keysrules:
    type: string
    regex: '(\w+)\.{1}(\w+)'
  valuesrules:
    type: dict
    matched: 'specs.repository'

cuda:
  type: dict
  nullable: false
  keysrules:
    type: string
    required: true
    regex: '(\d{1,2}\.\d{1}\.\d)?$'
  valuesrules:
    type: dict
    keysrules:
      type: string
      check_with: cudnn_threshold
    valuesrules:
      type: string

packages:
  type: dict
  keysrules:
    type: string
  valuesrules:
    type: dict
    allowed: [devel, cudnn, runtime]
    schema:
      devel:
        type: list
        dependencies: runtime
        check_with: available_dists
        schema:
          type: string
      runtime:
        required: true
        type: [string, list]
        check_with: available_dists
      cudnn:
        type: [string, list]
        check_with: available_dists
        dependencies: runtime

releases:
  type: dict
  keysrules:
    supported_dists: true
    type: string
  valuesrules:
    type: dict
    required: True
    matched: 'specs.releases'
"""  # noqa: W605, E501 # pylint: disable=W1401


class MetadataSpecValidator(CerberusValidator):
    """
    Custom cerberus validator for BentoML metadata spec.

    Refers to https://docs.python-cerberus.org/en/stable/customize.html#custom-rules.
    This will add two rules:
    - args should have the correct format: ARG=foobar as well as correct ENV format.
    - releases should be defined under SUPPORTED_OS_RELEASES

    Args:
        packages: bentoml release packages, bento-server
    """

    CUDNN_THRESHOLD: int = 1
    CUDNN_COUNTER: int = 0

    def __init__(self, *args: str, **kwargs: str) -> None:
        if "packages" in kwargs:
            self.packages = kwargs["packages"]
        if "releases" in kwargs:
            self.releases = kwargs["releases"]
        super(MetadataSpecValidator, self).__init__(*args, **kwargs)

    def _check_with_packages(self, field: str, value: str) -> None:
        """
        Check if registry is defined in packages.
        """
        if value not in self.packages:
            self._error(field, f"{field} is not defined under packages")

    def _check_with_args_format(self, field: str, value: "ValidateType") -> None:
        """
        Check if value has correct envars format: ARG=foobar
        This will be parsed at runtime when building docker images.
        """
        if isinstance(value, str):
            if "=" not in value:
                self._error(
                    field,
                    f"{value} should have format ARG=foobar",
                )
            else:
                envars, _, _ = value.partition("=")
                self._validate_env_vars(True, field, envars)
        if isinstance(value, list):
            for v in value:
                self._check_with_args_format(field, v)

    def _check_with_keys_format(self, field: str, value: str) -> None:
        fmt_field = [t[1] for t in string.Formatter().parse(value) if t[1] is not None]
        for _keys in self.document.keys():
            if _keys == field:
                continue
            if _keys not in fmt_field:
                self._error(
                    field, f"{value} doesn't contain {_keys} in defined document scope"
                )

    def _check_with_cudnn_threshold(self, field: str, value: str) -> None:
        if "cudnn" in value:
            self.CUDNN_COUNTER += 1
        if self.CUDNN_COUNTER > self.CUDNN_THRESHOLD:
            self._error(field, "Only allowed one CUDNN version per CUDA mapping")

    def _check_with_available_dists(self, field: str, value: "ValidateType") -> None:
        if isinstance(value, list):
            for v in value:
                self._check_with_available_dists(field, v)
        else:
            if value not in self.releases:
                self._error(field, f"{field} is not defined under releases")

    def _validate_supported_dists(
        self, supported_dists: bool, field: str, value: "ValidateType"
    ) -> None:
        """
        Validate if given is a supported OS.

        The rule's arguments are validated against this schema:
            {'type': 'boolean'}
        """
        if isinstance(value, str):
            if supported_dists and value not in SUPPORTED_OS_RELEASES:
                self._error(
                    field,
                    f"{value} is not defined in SUPPORTED_OS_RELEASES."
                    "If you are adding a new distros make sure "
                    "to add it to SUPPORTED_OS_RELEASES",
                )
        if isinstance(value, list):
            for v in value:
                self._validate_supported_dists(supported_dists, field, v)

    def _validate_env_vars(self, env_vars: bool, field: str, value: str) -> None:
        """
        Validate if given is a environment variable.

        The rule's arguments are validated against this schema:
            {'type': 'boolean'}
        """
        if isinstance(value, str):
            if env_vars and not value.isupper():
                self._error(field, f"{value} cannot be parsed to envars.")
        else:
            self._error(field, f"{value} cannot be parsed as string.")

    def _validate_matched(
        self,
        others: t.Union["ValidateType", "GenericDict"],
        field: str,
        value: t.Union["ValidateType", "GenericDict"],
    ) -> None:
        """
        Validate if a field is defined as a reference to match all given fields if
        we updated the default reference.
        The rule's arguments are validated against this schema:
            {'type': 'string'}
        """
        if isinstance(others, dict):
            if len(others) != len(value):
                self._error(
                    field,
                    f"type {type(others)} with ref "
                    f"{field} from {others} does not match",
                )
        elif isinstance(others, str):
            ref = functools.reduce(
                operator.getitem, others.split("."), self.root_document
            )
            if len(value) != len(ref):
                self._error(field, f"Reference {field} from {others} does not match")
        elif isinstance(others, list):
            for v in others:
                self._validate_matched(v, field, value)
        else:
            self._error(field, f"Unable to parse field type {type(others)}")


def validate_manifest_yaml(manifest: "GenericDict") -> bool:
    validated = False
    v = MetadataSpecValidator(yaml.safe_load(SPEC_SCHEMA))
    if not v.validate(manifest):
        v.clear_caches()
        logger.error(
            "[bold red]manifest is invalid. Errors as follow:[/bold red]",
            extra={"markup": True},
        )
        logger.exception(yaml.dump(v.errors, indent=2))
    else:
        logger.info(
            ":white_check_marks: [bold green]Valid manifest.[/bold green]",
            extra={"markup": True},
        )
        validated = True

    return validated


def add_validate_command(cli: click.Group) -> None:
    ...
