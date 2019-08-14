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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from sqlalchemy import Column, String, JSON, UniqueConstraint
from sqlalchemy.orm.exc import NoResultFound
from google.protobuf.json_format import MessageToDict, ParseDict

from bentoml.config import config
from bentoml.db import Base, create_session
from bentoml.proto import deployment_pb2


logger = logging.getLogger(__name__)


ALL_NAMESPACE_TAG = '__BENTOML_ALL_NAMESPACE'

class Deployment(Base):
    __tablename__ = 'deployments'
    __table_args__ = UniqueConstraint('name', 'namespace', name='_name_namespace_uc')

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    namespace = Column(String, nullable=False)

    spec = Column(JSON, nullable=False)
    labels = Column(JSON, nullable=False, default={})
    annotation = Column(JSON, nullable=False, default={})


def deployment_pb_to_orm_obj(deployment_pb):
    return Deployment(
        name=deployment_pb.name,
        namespace=deployment_pb.namespace,
        spec=MessageToDict(deployment_pb.spec),
        labels=MessageToDict(deployment_pb.labels),
        annotation=MessageToDict(deployment_pb.labels),
    )


def deployment_orm_obj_to_pb(deployment_obj):
    return deployment_pb2.Deployment(
        name=deployment_obj.name,
        namespace=deployment_obj.namespace,
        spec=ParseDict(deployment_obj.spec, deployment_pb2.DeploymentSpec()),
        labels=deployment_obj.labels,
        annotation=deployment_obj.annotation,
    )


class DeploymentStore(object):
    def __init__(self, sess_maker, default_namespace=None):
        self.sess_maker = sess_maker
        self.default_namespace = default_namespace or config.get(
            'deployment', 'default_namespace'
        )

    def add(self, deployment_pb):
        with create_session(self.sess_maker) as sess:
            deployment_obj = deployment_pb_to_orm_obj(deployment_pb)
            return sess.add(deployment_obj)

    def get(self, name, namespace=None):
        namespace = namespace or self.default_namespace
        with create_session(self.sess_maker) as sess:
            try:
                deployment_obj = (
                    sess.query(Deployment)
                    .filter_by(name=name, namespace=namespace)
                    .one()
                )
            except NoResultFound:
                return None

            return deployment_orm_obj_to_pb(deployment_obj)

    def delete(self, name, namespace=None):
        namespace = namespace or self.default_namespace
        with create_session(self.sess_maker) as sess:
            deployment = (
                sess.query(Deployment).filter_by(name=name, namespace=namespace).one()
            )
            return sess.remove(deployment)

    def list(
        self, namespace=None, filter_str=None, labels=None, offset=None, limit=None
    ):
        namespace = namespace or self.default_namespace
        with create_session(self.sess_maker) as sess:
            query = sess.query(Deployment)
            if namespace != ALL_NAMESPACE_TAG: # else query all namespaces
                query.filter_by(namespace=namespace)
            if limit:
                query.limit(limit)
            if offset:
                query.offset(offset)
            if filter_str:
                query.filter(Deployment.name.contains(filter_str))
            if labels:
                raise NotImplementedError("Listing by labels is not yet implemented")
            return query.all()
