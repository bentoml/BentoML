from sqlalchemy import UniqueConstraint, Column, Integer, String, JSON, and_, or_

from bentoml.exceptions import YataiServiceException
from bentoml.yatai.db import Base
from bentoml.yatai.proto.label_selectors_pb2 import LabelSelectors


class Labels(Base):
    __tablename__ = 'labels'
    __table_args__ = tuple(
        UniqueConstraint(
            'resource_type',
            'resource_id',
            'key',
            name='_resource_type_resource_id_key_uc',
        )
    )
    id = Column(Integer, primary_key=True)
    resource_type = Column(String, nullable=False)
    resource_id = Column(Integer, nullable=False)
    key = Column(String, nullable=False)
    value = Column(String, nullable=False)


def filter_label_query(sess, resource_type, label_selectors):
    query = sess.query(Labels.resource_id)
    query = query.filter_by(resource_type=resource_type)
    filters = []
    for key in label_selectors.match_labels:
        filters.append(
            and_(Labels.key == key, Labels.value == label_selectors.match_labels[key])
        )
    for expression in label_selectors.match_expressions:
        if (
            expression.operator
            == LabelSelectors.LabelSelectorExpression.OPERATOR_TYPE.In
        ):
            filters.append(
                and_(Labels.key == expression.key, Labels.value.in_(expression.values))
            )
        elif (
            expression.operator
            == LabelSelectors.LabelSelectorExpression.OPERATOR_TYPE.NotIn
        ):
            filters.append(
                and_(Labels.key == expression.key, ~Labels.value.in_(expression.values))
            )
        elif (
            expression.operator
            == LabelSelectors.LabelSelectorExpression.OPERATOR_TYPE.Exists
        ):
            filters.append(Labels.key == expression.key)
        elif (
            expression.operator
            == LabelSelectors.LabelSelectorExpression.OPERATOR_TYPE.DoesNotExist
        ):
            filters.append(Labels.key != expression.key)
        else:
            raise YataiServiceException(
                f'Unrecognized operator: "{expression.operator}"'
            )
    query = query.filter(or_(*filters))
    result = query.all()
    return [row[0] for row in result]
