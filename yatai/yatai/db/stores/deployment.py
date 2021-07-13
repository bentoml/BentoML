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

import logging
import datetime
from contextlib import contextmanager

from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    JSON,
    UniqueConstraint,
    desc,
)
from sqlalchemy.orm.exc import NoResultFound
from google.protobuf.json_format import ParseDict

from bentoml.exceptions import YataiDeploymentException
from bentoml.yatai.deployment import ALL_NAMESPACE_TAG
from bentoml.yatai.db.base import Base
from bentoml.yatai.db.stores.label import (
    LabelStore,
    RESOURCE_TYPE,
)
from bentoml.yatai.proto import deployment_pb2
from bentoml.yatai.proto.deployment_pb2 import DeploymentSpec, ListDeploymentsRequest
from bentoml.utils import ProtoMessageToDict

logger = logging.getLogger(__name__)


class Deployment(Base):
    __tablename__ = 'deployments'
    __table_args__ = tuple(
        UniqueConstraint('name', 'namespace', name='_name_namespace_uc')
    )

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    namespace = Column(String, nullable=False)

    spec = Column(JSON, nullable=False, default={})
    state = Column(JSON, nullable=False, default={})
    annotations = Column(JSON, nullable=False, default={})

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_updated_at = Column(DateTime, default=datetime.datetime.utcnow)


def _deployment_pb_to_orm_obj(deployment_pb, deployment_obj=Deployment()):
    deployment_obj.name = deployment_pb.name
    deployment_obj.namespace = deployment_pb.namespace
    deployment_obj.spec = ProtoMessageToDict(deployment_pb.spec)
    deployment_obj.state = ProtoMessageToDict(deployment_pb.state)
    deployment_obj.annotations = dict(deployment_pb.annotations)
    deployment_obj.created_at = deployment_pb.created_at.ToDatetime()
    deployment_obj.last_updated_at = deployment_pb.last_updated_at.ToDatetime()
    return deployment_obj


def _deployment_orm_obj_to_pb(deployment_obj, labels=None):
    deployment_pb = deployment_pb2.Deployment(
        name=deployment_obj.name,
        namespace=deployment_obj.namespace,
        spec=ParseDict(deployment_obj.spec, deployment_pb2.DeploymentSpec()),
        state=ParseDict(deployment_obj.state, deployment_pb2.DeploymentState()),
        annotations=deployment_obj.annotations,
    )
    deployment_pb.created_at.FromDatetime(deployment_obj.created_at)
    if deployment_obj.last_updated_at:
        deployment_pb.last_updated_at.FromDatetime(deployment_obj.last_updated_at)
    if labels is not None:
        deployment_pb.labels.update(labels)
    return deployment_pb


class DeploymentStore(object):
    @staticmethod
    def insert_or_update(sess, deployment_pb):
        try:
            deployment_obj = (
                sess.query(Deployment)
                .filter_by(name=deployment_pb.name, namespace=deployment_pb.namespace)
                .one()
            )
            if deployment_obj:
                # updating deployment record in db
                _deployment_pb_to_orm_obj(deployment_pb, deployment_obj)
                if deployment_pb.labels:
                    LabelStore.add_or_update(
                        sess,
                        RESOURCE_TYPE.deployment,
                        deployment_obj.id,
                        deployment_pb.labels,
                    )
        except NoResultFound:
            deployment_orm_obj = _deployment_pb_to_orm_obj(deployment_pb)
            sess.add(deployment_orm_obj)
            if deployment_pb.labels:
                deployment_row = (
                    sess.query(Deployment)
                    .filter_by(
                        name=deployment_orm_obj.name,
                        namespace=deployment_orm_obj.namespace,
                    )
                    .one()
                )
                LabelStore.add(
                    sess,
                    RESOURCE_TYPE.deployment,
                    deployment_row.id,
                    deployment_pb.labels,
                )

    @staticmethod
    @contextmanager
    def update_deployment(sess, name, namespace):
        try:
            deployment_obj = (
                sess.query(Deployment).filter_by(name=name, namespace=namespace).one()
            )
            yield deployment_obj
        except NoResultFound:
            yield None

    @staticmethod
    def get(sess, name, namespace):
        try:
            deployment_obj = (
                sess.query(Deployment).filter_by(name=name, namespace=namespace).one()
            )
            labels = LabelStore.get(sess, RESOURCE_TYPE.deployment, deployment_obj.id)
        except NoResultFound:
            return None

        return _deployment_orm_obj_to_pb(deployment_obj, labels)

    @staticmethod
    def delete(sess, name, namespace):
        try:
            deployment = (
                sess.query(Deployment).filter_by(name=name, namespace=namespace).one()
            )
            LabelStore.delete(
                sess, resource_type=RESOURCE_TYPE.deployment, resource_id=deployment.id,
            )
            return sess.delete(deployment)
        except NoResultFound:
            raise YataiDeploymentException(
                "Deployment '%s' in namespace: '%s' is not found" % name, namespace
            )

    @staticmethod
    def list(
        sess,
        namespace,
        operator=None,
        label_selectors=None,
        offset=None,
        limit=None,
        order_by=ListDeploymentsRequest.created_at,
        ascending_order=False,
    ):
        query = sess.query(Deployment)
        order_by = ListDeploymentsRequest.SORTABLE_COLUMN.Name(order_by)
        order_by_field = getattr(Deployment, order_by)
        order_by_action = order_by_field if ascending_order else desc(order_by_field)
        query = query.order_by(order_by_action)
        if label_selectors.match_labels or label_selectors.match_expressions:
            deployment_ids = LabelStore.filter_query(
                sess, RESOURCE_TYPE.deployment, label_selectors
            )
            query.filter(Deployment.id.in_(deployment_ids))
        if namespace != ALL_NAMESPACE_TAG:  # else query all namespaces
            query = query.filter_by(namespace=namespace)
        if operator:
            operator_name = DeploymentSpec.DeploymentOperator.Name(operator)
            query = query.filter(
                Deployment.spec['operator'].as_string().contains(operator_name)
            )
        # We are not defaulting limit to 200 in the signature,
        # because protobuf will pass 0 as value
        limit = limit or 200
        # Limit and offset need to be called after order_by filter/filter_by is
        # called
        query = query.limit(limit)
        if offset:
            query = query.offset(offset)
        query_result = query.all()
        deployment_ids = [deployment_obj.id for deployment_obj in query_result]
        labels = LabelStore.list(sess, RESOURCE_TYPE.deployment, deployment_ids)

        return [
            _deployment_orm_obj_to_pb(
                deployment_obj, labels.get(str(deployment_obj.id))
            )
            for deployment_obj in query_result
        ]
