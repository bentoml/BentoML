import re

from bentoml.exceptions import InvalidArgument

dns1123_label_fmt = "[a-z0-9]([-a-z0-9]*[a-z0-9])?"
dns1123_subdomain_max_length = 253
dns1123_subdomain_max_length_error_msg = (
    "a valid DNS1123 subbdomain name must be less than "
    f"{dns1123_subdomain_max_length} characters in length"
)
dns1123_subdomain_fmt = dns1123_label_fmt + "(\\." + dns1123_label_fmt + ")*"
dns1123_subdomain_error_msg = (
    "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters"
    ", '-' or '.', and must start and end with an alphanumeric character"
)
dns1123_subdomain_regex = re.compile(f"^{dns1123_subdomain_fmt}$")


def check_is_dns1123_subdomain(value: str):
    """tests if a string conforms to the definition of a subdomain in DNS (RFC 1123)"""
    errors = []
    if len(value) > dns1123_subdomain_max_length:
        errors.append(dns1123_subdomain_max_length_error_msg)
    if dns1123_subdomain_regex.match(value) is None:
        errors.append(dns1123_subdomain_error_msg)

    if errors:
        raise InvalidArgument(",".join(errors))
