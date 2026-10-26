import pytest

from bentoml._internal.utils.dotenv import parse_dotenv


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("$A:$AB", "short:long"),
        (r"\$A:$A", "$A:short"),
        (r"\${A}:${A}", "${A}:short"),
        ("${A}:$AB", "short:long"),
        ("$AB:$A:$AB", "long:short:long"),
        ('"$A:$AB"', "short:long"),
        ("'$A:$AB'", "$A:$AB"),
    ],
)
def test_substitute_each_variable_once(value: str, expected: str) -> None:
    env = parse_dotenv(f"A=short\nAB=long\nRESULT={value}")
    assert env["RESULT"] == expected


def test_do_not_expand_variables_in_replacement() -> None:
    env = parse_dotenv("A='$AB'\nAB=long\nRESULT=$A:$AB")
    assert env["RESULT"] == "$AB:long"
