import re

from bentoml.exceptions import InvalidArgument

dns1123_label_fmt = "[_a-z0-9]([-_a-z0-9]*[_a-z0-9])?"
dns1123_subdomain_max_length = 253
dns1123_subdomain_max_length_error_msg = (
    "a tag must be less than {dns1123_subdomain_max_length} characters in length"
)
dns1123_subdomain_fmt = dns1123_label_fmt + "(\\." + dns1123_label_fmt + ")*"
dns1123_subdomain_error_msg = "a tag must consist of alphanumeric characters, '_', '-', or '.', and must not start or end with '-'"
dns1123_subdomain_regex = re.compile(f"^{dns1123_subdomain_fmt}$")


def validate_tag_name_str(value: str):
    """tests if a tag string conforms to the definition of a subdomain in DNS (RFC 1123)"""
    errors = []
    if len(value) > dns1123_subdomain_max_length:
        errors.append(dns1123_subdomain_max_length_error_msg)
    if dns1123_subdomain_regex.match(value) is None:
        errors.append(dns1123_subdomain_error_msg)

    if errors:
        raise InvalidArgument(", and ".join(errors))


def validate_version_str(version_str):
    """
    Validate that version str format is either a simple version string that:
        * Consist of only ALPHA / DIGIT / "-" / "." / "_"
        * Length between 1-128
    Or a valid semantic version https://github.com/semver/semver/blob/master/semver.md
    """
    regex = r"[A-Za-z0-9_.-]{1,128}\Z"
    semver_regex = r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"  # noqa: E501
    if (
        re.match(regex, version_str) is None
        and re.match(semver_regex, version_str) is None
    ):
        raise InvalidArgument(
            'Invalid Service version: "{}", it can only consist'
            ' ALPHA / DIGIT / "-" / "." / "_", and must be less than'
            "128 characters".format(version_str)
        )

    if version_str.lower() == "latest":
        raise InvalidArgument('Service version can not be set to "latest"')
