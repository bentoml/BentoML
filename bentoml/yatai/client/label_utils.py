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


def expressions_extractor(query):
    # inline comments
    return re.split(r',(?![^()]*\))', query)


def value_string_to_list(value_string):
    if not value_string.startswith('(') or not value_string.endswith(')'):
        raise BentoMLException(
            "Query values need to be inside (). (e.g. (value1, value2, ..)"
        )
    return [value.strip() for value in value_string[1:-1].split(',')]


def expression_element_extractor(expression):
    return re.split(r' (?![^()]*\))', expression)


def generate_gprc_labels_selector(label_selectors, label_query):
    if not label_query:
        return
    expressions = expressions_extractor(label_query)
    for expression in expressions:
        expression = expression.strip()
        elements = expression_element_extractor(expression.strip())
        if len(elements) == 1:
            # Possible queries: key=value, key!=value, key
            query = elements[0].strip()
            if query.lower() in [i.lower() for i in label_expression_operators.keys()]:
                # TODO better message
                raise BentoMLException("Label query operator can't be the only element")
            if '!=' in query:
                key, value = query.split('!=')
                label_selectors.match_expressions.append(
                    LabelSelectors.LabelSelectorExpression(
                        key=key,
                        operator=label_expression_operators['DoesNotExist'],
                        values=[value],
                    )
                )
            elif '=' in query:
                key, value = query.split('=')
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
            assert operator.lower in [
                i.lower for i in ['Exists', 'DoesNotExist']
            ], 'Operator is invalid'
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
                i.lower for i in ['In', 'NotIn']
            ], 'Operator is invalid'
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
