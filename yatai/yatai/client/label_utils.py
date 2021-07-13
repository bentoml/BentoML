# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re

from bentoml.exceptions import BentoMLException
from bentoml.yatai.proto.label_selectors_pb2 import LabelSelectors


label_expression_operators = {
    i[0]: i[1] for i in LabelSelectors.LabelSelectorExpression.OPERATOR_TYPE.items()
}


def _extract_expressions(query):
    # Using regex to split query base on ",".  negative lookahead (?!...) / don't match
    # anything inside the ().
    # e.g.
    # "key, key1 in (value1, value2),key3"
    # -> ['key', ' key1 in (value1, value2)', 'key3']
    return re.split(r',(?![^()]*\))', query)


def value_string_to_list(value_string):
    if not value_string.startswith('(') or not value_string.endswith(')'):
        raise BentoMLException(
            f"Query values {value_string} need to be inside (). "
            f"e.g. (value1, value2, ..)"
        )
    if len(value_string) == 2:
        raise BentoMLException("Query values can't be empty")
    return [value.strip() for value in value_string[1:-1].split(',')]


def _extract_expression_elements(expression):
    # Using regex to split query base on " ".  negative lookahead (?!...) / don't match
    # anything inside the ().
    # e.g.
    # "key1 in (value1, value2,value4)"
    # -> ['key1', ' in', '(value1, value2,,value3)']
    return re.split(r' (?![^()]*\))', expression)


def generate_gprc_labels_selector(label_selectors, label_query):
    """Parse label query string and update to the label selector request"""
    if not label_query:
        return
    expressions = _extract_expressions(label_query)
    for expression in expressions:
        expression = expression.strip()
        elements = _extract_expression_elements(expression.strip())
        if len(elements) == 1:
            # Possible queries: key=value, key!=value, key
            query = elements[0].strip()
            if query.lower() in [i.lower() for i in label_expression_operators.keys()]:
                raise BentoMLException(
                    f"Label query operator {query} can't be the only element"
                )
            if '!=' in query:
                if query.count('!=') > 1:
                    raise BentoMLException(f"Too many '!=' operator in query {query}")
                key, value = query.split('!=')
                if not value:
                    raise BentoMLException(f"Label {query} can't have empty value")
                label_selectors.match_expressions.append(
                    LabelSelectors.LabelSelectorExpression(
                        key=key,
                        operator=label_expression_operators['NotIn'],
                        values=[value],
                    )
                )
            elif '=' in query:
                if query.count('=') > 1:
                    raise BentoMLException(f"Too many '=' operator in query {query}")
                key, value = query.split('=')
                if not value:
                    raise BentoMLException(f"Label {query} can't have empty value")
                label_selectors.match_labels[key] = value
            else:
                label_selectors.match_expressions.append(
                    LabelSelectors.LabelSelectorExpression(
                        key=query, operator=label_expression_operators['Exists']
                    )
                )
        elif len(elements) == 2:
            # possible queries: key Exists/exists, key DoesNotExist/doesnotexist
            key = elements[0].strip()
            operator = elements[1].strip()
            assert operator.lower() in [
                i.lower() for i in ['Exists', 'DoesNotExist']
            ], f'Operator "{operator}" is invalid'
            operator = 'Exists' if operator.capitalize() == 'Exists' else 'DoesNotExist'
            label_selectors.match_expressions.append(
                LabelSelectors.LabelSelectorExpression(
                    key=key, operator=label_expression_operators[operator],
                )
            )
        elif len(elements) == 3:
            # possible queries: key In/in (values), key NotIn/notin (values)
            key = elements[0].strip()
            operator = elements[1].strip()
            assert operator.lower() in [
                i.lower() for i in ['In', 'NotIn']
            ], f'Operator "{operator}" is invalid'
            operator = 'In' if operator.capitalize() == 'In' else 'NotIn'
            values = value_string_to_list(elements[2].strip())
            label_selectors.match_expressions.append(
                LabelSelectors.LabelSelectorExpression(
                    key=key,
                    operator=label_expression_operators[operator],
                    values=values,
                )
            )
        else:
            raise BentoMLException(f'Too many elements in the label query {expression}')
    return
