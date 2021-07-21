import enum
import re

from sqlalchemy import UniqueConstraint, Column, Integer, String, and_, Enum

from bentoml.exceptions import YataiLabelException, InvalidArgument
from bentoml.yatai.db import Base
from bentoml.yatai.proto.label_selectors_pb2 import LabelSelectors


class RESOURCE_TYPE(enum.Enum):
    deployment = 1
    bento = 2


def _validate_labels(labels):
    """
    Validate labels key value format is:
        * Between 3 and 63 characters
        * Consist of alphanumeric, dash (-), period (.), and underscore (_)
        * Start and end with alphanumeric
    Args:
        labels: Dictionary

    Returns:
    Raise:
        InvalidArgument
    """
    if not isinstance(labels, dict):
        raise InvalidArgument('BentoService labels must be a dictionary')

    pattern = re.compile("^(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?$")
    for key in labels:
        if (
            not (63 >= len(key) >= 3)
            or not (63 >= len(labels[key]) >= 3)
            or not pattern.match(key)
            or not pattern.match(labels[key])
        ):
            raise InvalidArgument(
                f'Invalide label {key}:{labels[key]}. Valid label key and value must '
                f'be between 3 to 63 characters and must be begin and end with '
                f'an alphanumeric character ([a-z0-9A-Z]) with dashes (-), '
                f'underscores (_), and dots (.).'
            )


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
    resource_type = Column(Enum(RESOURCE_TYPE))
    resource_id = Column(Integer, nullable=False)
    key = Column(String, nullable=False)
    value = Column(String, nullable=False)


class LabelStore(object):
    @staticmethod
    def add(sess, resource_type, resource_id, labels):
        label_objects = []
        for key in labels:
            label_obj = Label()
            label_obj.resource_type = resource_type
            label_obj.resource_id = resource_id
            label_obj.key = key
            label_obj.value = labels[key]
            label_objects.append(label_obj)
        return sess.add_all(label_objects)

    @staticmethod
    def add_or_update(sess, resource_type, resource_id, labels):
        label_rows = (
            sess.query(Label)
            .filter(
                Label.resource_type == resource_type, Label.resource_id == resource_id
            )
            .all()
        )
        if len(label_rows) == 0:
            return LabelStore.add(sess, resource_type, resource_id, labels)
        else:
            for row in label_rows:
                if labels.get(row.key, None) is not None:
                    row.value = labels[row.key]

    @staticmethod
    def get(sess, resource_type, resource_id):
        labels = (
            sess.query(Label)
            .filter_by(resource_type=resource_type, resource_id=resource_id)
            .all()
        )
        return {row.key: row.value for row in labels}

    @staticmethod
    def list(sess, resource_type, resource_ids):
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
            if not labels.get(str(label.resource_id)):
                labels[str(label.resource_id)] = {}
            labels[str(label.resource_id)][label.key] = label.value
        return labels

    @staticmethod
    def delete(sess, resource_type, resource_id, labels=None):
        filters = [
            Label.resource_id == resource_id,
            Label.resource_type == resource_type,
        ]
        if labels is not None:
            filters.append(Label.key.in_(list(labels.keys())))
        return sess.query(Label).filter(and_(*filters)).delete()

    @staticmethod
    def filter_query(sess, resource_type, label_selectors):
        query = sess.query(Label.resource_id)
        query = query.filter_by(resource_type=resource_type)
        filters = []
        for key in label_selectors.match_labels:
            filters.append(
                sess.query(Label.resource_id).filter(
                    and_(
                        Label.key == key,
                        Label.value == label_selectors.match_labels[key],
                    )
                )
            )
        for expression in label_selectors.match_expressions:
            if (
                expression.operator
                == LabelSelectors.LabelSelectorExpression.OPERATOR_TYPE.In
            ):
                filters.append(
                    sess.query(Label.resource_id).filter(
                        and_(
                            Label.key == expression.key,
                            Label.value.in_(expression.values),
                        )
                    )
                )
            elif (
                expression.operator
                == LabelSelectors.LabelSelectorExpression.OPERATOR_TYPE.NotIn
            ):
                filters.append(
                    sess.query(Label.resource_id).filter(
                        and_(
                            Label.key == expression.key,
                            ~Label.value.in_(expression.values),
                        )
                    )
                )
            elif (
                expression.operator
                == LabelSelectors.LabelSelectorExpression.OPERATOR_TYPE.Exists
            ):
                filters.append(
                    sess.query(Label.resource_id).filter(Label.key == expression.key)
                )
            elif (
                expression.operator
                == LabelSelectors.LabelSelectorExpression.OPERATOR_TYPE.DoesNotExist
            ):
                filters.append(
                    sess.query(Label.resource_id).filter(Label.key != expression.key)
                )
            else:
                raise YataiLabelException(
                    f'Unrecognized operator: "{expression.operator}"'
                )
        query = query.intersect(*filters)
        result = query.all()
        return [row[0] for row in result]
