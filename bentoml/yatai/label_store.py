from sqlalchemy import UniqueConstraint, Column, Integer, String, and_, or_

from bentoml.exceptions import YataiLabelException
from bentoml.yatai.db import Base
from bentoml.yatai.proto.label_selectors_pb2 import LabelSelectors


class Label(Base):
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


def add_labels(sess, resource_type, resource_id, labels):
    label_objects = []
    for key in labels:
        label_obj = Label()
        label_obj.resource_type = resource_type
        label_obj.resource_id = resource_id
        label_obj.key = key
        label_obj.value = labels[key]
        label_objects.append(label_obj)
    return sess.add_all(label_objects)


def add_or_update_labels(sess, resource_type, resource_id, labels):
    label_rows = (
        sess.query(Label)
        .filter(Label.resource_type == resource_type, Label.resource_id == resource_id)
        .all()
    )
    if len(label_rows) == 0:
        return add_labels(sess, resource_type, resource_id, labels)
    else:
        for row in label_rows:
            if labels.get(row.key, None) is not None:
                row.value = labels[row.key]


def get_labels(sess, resource_type, resource_id):
    labels = (
        sess.query(Label)
        .filter_by(resource_type=resource_type, resource_id=resource_id)
        .all()
    )
    return {row.key: row.value for row in labels}


def list_labels(sess, resource_type, resource_ids):
    label_rows = (
        sess.query(Label)
        .filter(
            and_(
                Label.resource_type == resource_type,
                Label.resource_id.in_(resource_ids),
            )
        )
        .all()
    )
    labels = {}
    for label in label_rows:
        if labels.get(str(label.resource_id), None) is None:
            labels[str(label.resource_id)] = {}
        labels[str(label.resource_id)][label.key] = label.value
    return labels


def delete_labels(sess, resource_type, resource_id, labels=None):
    filters = [Label.resource_id == resource_id, Label.resource_type == resource_type]
    if labels is not None:
        filters.append(Label.key.in_(list(labels.keys())))
    return sess.query(Label).filter(and_(*filters)).delete()


def filter_label_query(sess, resource_type, label_selectors):
    query = sess.query(Label.resource_id)
    query = query.filter_by(resource_type=resource_type)
    filters = []
    for key in label_selectors.match_labels:
        filters.append(
            and_(Label.key == key, Label.value == label_selectors.match_labels[key])
        )
    for expression in label_selectors.match_expressions:
        if (
            expression.operator
            == LabelSelectors.LabelSelectorExpression.OPERATOR_TYPE.In
        ):
            filters.append(
                and_(Label.key == expression.key, Label.value.in_(expression.values))
            )
        elif (
            expression.operator
            == LabelSelectors.LabelSelectorExpression.OPERATOR_TYPE.NotIn
        ):
            filters.append(
                and_(Label.key == expression.key, ~Label.value.in_(expression.values))
            )
        elif (
            expression.operator
            == LabelSelectors.LabelSelectorExpression.OPERATOR_TYPE.Exists
        ):
            filters.append(Label.key == expression.key)
        elif (
            expression.operator
            == LabelSelectors.LabelSelectorExpression.OPERATOR_TYPE.DoesNotExist
        ):
            filters.append(Label.key != expression.key)
        else:
            raise YataiLabelException(f'Unrecognized operator: "{expression.operator}"')
    query = query.filter(or_(*filters))
    result = query.all()
    return [row[0] for row in result]
