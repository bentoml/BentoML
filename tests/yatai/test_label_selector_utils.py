import pytest

from bentoml.exceptions import BentoMLException
from bentoml.yatai.client.label_utils import (
    _extract_expressions,
    _extract_expression_elements,
    generate_gprc_labels_selector,
)
from bentoml.yatai.proto.label_selectors_pb2 import LabelSelectors


def test_expressions_extractor_func():
    one_expression_query = "test!=true"
    assert len(_extract_expressions(one_expression_query)) == 1, "something"

    one_expression_with_parentheses = "(a,b,c,,)"
    assert len(_extract_expressions(one_expression_with_parentheses)) == 1, "something"

    multiple_expressions_with_parentheses = "abc, foo,(bar, value), value2"
    assert (
        len(_extract_expressions(multiple_expressions_with_parentheses)) == 4
    ), "something"


def test_expression_element_extractor():
    one_element_query = "abdas,"
    assert len(_extract_expression_elements(one_element_query)) == 1, ""

    one_element_with_parentheses = "(abb, bbb, cbb)"
    assert len(_extract_expression_elements(one_element_with_parentheses)) == 1, ""

    multiple_element_with_parentheses = "foo bar (abc, efg g),f fourth"
    assert len(_extract_expression_elements(multiple_element_with_parentheses)) == 4, ""


def test_generate_grpc_labels_selector():
    success_single_query = "Test"
    success_single_selector = LabelSelectors()
    generate_gprc_labels_selector(success_single_selector, success_single_query)
    assert len(success_single_selector.match_expressions) == 1, ""
    assert success_single_selector.match_expressions[0].key == "Test"
    assert (
        success_single_selector.match_expressions[0].operator
        == LabelSelectors.LabelSelectorExpression.OPERATOR_TYPE.Exists
    ), ""

    success_multiple_query = "Key1,key2 in (value1, value2), key3 Exists"
    success_multiple_selector = LabelSelectors()
    generate_gprc_labels_selector(success_multiple_selector, success_multiple_query)
    assert len(success_multiple_selector.match_expressions) == 3, ""
    assert success_multiple_selector.match_expressions[1].key == "key2", ""
    assert success_multiple_selector.match_expressions[1].values == [
        "value1",
        "value2",
    ], ""

    success_single_label_query = "foo=bar"
    success_single_label_selector = LabelSelectors()
    generate_gprc_labels_selector(
        success_single_label_selector, success_single_label_query
    )
    assert "foo" in success_single_label_selector.match_labels.keys(), ""
    assert "bar" == success_single_label_selector.match_labels["foo"], ""
    assert len(success_single_label_selector.match_labels.keys()) == 1, ""

    success_multiple_labels_query = "foo=bar,test=pass, foo=replaced"
    success_multiple_labels_selector = LabelSelectors()
    generate_gprc_labels_selector(
        success_multiple_labels_selector, success_multiple_labels_query
    )
    assert len(success_multiple_labels_selector.match_labels.keys()) == 2, ""
    assert success_multiple_labels_selector.match_labels["foo"] == "replaced", ""

    success_mixed_query = (
        "foo=bar, foo=replaced,test=passed,key1 in (value1, value2), "
        "key2 Exists, key3, user!=admin"
    )
    success_mixed_selector = LabelSelectors()
    generate_gprc_labels_selector(success_mixed_selector, success_mixed_query)
    assert len(success_mixed_selector.match_expressions) == 4, ""
    assert success_mixed_selector.match_labels["foo"] == "replaced", ""
    assert "user" not in success_mixed_selector.match_labels.keys(), ""
    assert success_mixed_selector.match_expressions[3].key == "user", ""
    assert success_mixed_selector.match_expressions[3].values == ["admin"], ""
    assert not success_mixed_selector.match_expressions[1].values, ""

    with pytest.raises(BentoMLException) as e:
        generate_gprc_labels_selector(LabelSelectors(), "key=value1=value2")
    assert str(e.value).startswith("Too many '=' operator in")

    with pytest.raises(BentoMLException) as e:
        generate_gprc_labels_selector(LabelSelectors(), "In")
    assert str(e.value).startswith("Label query operator In can't be the only element")

    with pytest.raises(BentoMLException) as e:
        generate_gprc_labels_selector(LabelSelectors(), "key in value1 value2")
    assert str(e.value).startswith("Too many elements in the label query")

    with pytest.raises(AssertionError) as e:
        generate_gprc_labels_selector(LabelSelectors(), "Key IncorrectOperator")
    assert str(e.value).startswith('Operator "IncorrectOperator" is invalid')

    with pytest.raises(AssertionError) as e:
        generate_gprc_labels_selector(LabelSelectors(), "Key ins (value)")
    assert str(e.value).startswith('Operator "ins" is invalid')

    with pytest.raises(BentoMLException) as e:
        generate_gprc_labels_selector(LabelSelectors(), "key In ()")
    assert str(e.value).startswith("Query values can't be empty")
