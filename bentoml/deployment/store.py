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

from sqlalchemy import Column, String, JSON
from google.protobuf.json_format import MessageToDict

from bentoml.db import Base, create_session


class Deployment(Base):
    __tablename__ = 'deployments'

    id = Column(String, primary_key=True)
    name = Column(String, unique=True)
    namespace = Column(String)

    spec = Column(JSON)
    labels = Column(JSON)
    annotation = Column(JSON)


class DeploymentStore(object):
    def __init__(self):
        pass

    def add(self, deployment_pb):
        with create_session() as session:
            deployment_obj = Deployment(
                name=deployment_pb.name,
                namespace=deployment_pb.namespace,
                spec=MessageToDict(deployment_pb.spec),
                labels=MessageToDict(deployment_pb.labels),
                annotation=MessageToDict(deployment_pb.labels),
            )
            session.add(deployment_obj)

    def get(self, name):
        with create_session() as session:
            return session.query(Deployment).filter_by(name=name).first()

    def delete(self, name):
        pass

    def list(self, filter_str, labels, offset, limit):
        pass
