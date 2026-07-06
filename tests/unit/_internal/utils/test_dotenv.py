from bentoml._internal.utils.dotenv import parse_dotenv


def test_simple_assignment():
    assert parse_dotenv("KEY=value") == {"KEY": "value"}


def test_export_prefix_is_stripped():
    assert parse_dotenv("export KEY=value") == {"KEY": "value"}


def test_colon_separator():
    assert parse_dotenv("KEY: value") == {"KEY": "value"}


def test_double_quoted_value():
    assert parse_dotenv('KEY="value"') == {"KEY": "value"}


def test_single_quoted_value():
    assert parse_dotenv("KEY='value'") == {"KEY": "value"}


def test_empty_value():
    assert parse_dotenv("KEY=") == {"KEY": ""}


def test_trailing_comment_is_stripped():
    assert parse_dotenv("KEY=value # trailing comment") == {"KEY": "value"}


def test_comment_and_blank_lines_are_ignored():
    assert parse_dotenv("# a comment\n\nKEY=v") == {"KEY": "v"}


def test_variable_substitution():
    assert parse_dotenv("A=1\nB=${A}x") == {"A": "1", "B": "1x"}


def test_escaped_variable_is_not_substituted():
    # A leading backslash escapes the reference, leaving it literal.
    assert parse_dotenv("A=1\nB=\\$A") == {"A": "1", "B": "$A"}


def test_single_quotes_disable_substitution():
    assert parse_dotenv("A=1\nB='$A'") == {"A": "1", "B": "$A"}
