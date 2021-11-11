import re

from ...exceptions import InvalidArgument

tag_fmt = "[a-z0-9]([-._a-z0-9]*[a-z0-9])?"
tag_max_length = 63
tag_max_length_error_msg = (
    "a tag's name or version must be at most {tag_max_length} characters in length"
)
tag_invalid_error_msg = "a tag's name or version must consist of alphanumeric characters, '_', '-', or '.', and must start and end with an alphanumeric character"
tag_regex = re.compile(f"^{tag_fmt}$")


def validate_tag_str(value: str):
    """
    Validate that a tag value (either name or version) is a simple string that:
        * Must be at most 63 characters long,
        * Begin and end with an alphanumeric character (`[a-z0-9A-Z]`), and
        * May contain dashes (`-`), underscores (`_`) dots (`.`), or alphanumerics
          between.
    """
    errors = []
    if len(value) > tag_max_length:
        errors.append(tag_max_length_error_msg)
    if tag_regex.match(value) is None:
        errors.append(tag_invalid_error_msg)

    if errors:
        raise InvalidArgument(f"{value} is not a valid tag: " + ", and ".join(errors))
