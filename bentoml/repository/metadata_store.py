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
import datetime

from sqlalchemy import (
    Column,
    Enum,
    String,
    Integer,
    JSON,
    Boolean,
    DateTime,
    UniqueConstraint,
    desc,
)
from sqlalchemy.orm.exc import NoResultFound
from google.protobuf.json_format import ParseDict

from bentoml.utils import ProtoMessageToDict
from bentoml.exceptions import YataiRepositoryException
from bentoml.db import Base, create_session
from bentoml.proto.repository_pb2 import (
    UploadStatus,
    BentoUri,
    BentoServiceMetadata,
    Bento as BentoPB,
    ListBentoRequest,
)

logger = logging.getLogger(__name__)


DEFAULT_UPLOAD_STATUS = UploadStatus(status=UploadStatus.UNINITIALIZED)


class Bento(Base):
    __tablename__ = 'bentos'
    __table_args__ = tuple(UniqueConstraint('name', 'version', name='_name_version_uc'))

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)

    # Storage URI for this Bento
    uri = Column(String, nullable=False)
    uri_type = Column(Enum(*BentoUri.StorageType.keys()), default=BentoUri.UNSET)

    # JSON filed mapping directly to BentoServiceMetadata proto message
    bento_service_metadata = Column(JSON, nullable=False, default={})

    # Time of AddBento call, the time of Bento creation can be found in metadata field
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # latest upload status, JSON message also includes last update timestamp
    upload_status = Column(
        JSON, nullable=False, default=ProtoMessageToDict(DEFAULT_UPLOAD_STATUS)
    )

    # mark as deleted
    deleted = Column(Boolean, default=False)


def _bento_orm_obj_to_pb(bento_obj):
    bento_service_metadata_pb = ParseDict(
        bento_obj.bento_service_metadata, BentoServiceMetadata()
    )
    bento_uri = BentoUri(
        uri=bento_obj.uri, type=BentoUri.StorageType.Value(bento_obj.uri_type)
    )
    return BentoPB(
        name=bento_obj.name,
        version=bento_obj.version,
        uri=bento_uri,
        bento_service_metadata=bento_service_metadata_pb,
    )


class BentoMetadataStore(object):
    def __init__(self, sess_maker):
        self.sess_maker = sess_maker

    def add(self, bento_name, bento_version, uri, uri_type):
        with create_session(self.sess_maker) as sess:
            bento_obj = Bento()
            bento_obj.name = bento_name
            bento_obj.version = bento_version
            bento_obj.uri = uri
            bento_obj.uri_type = BentoUri.StorageType.Name(uri_type)
            return sess.add(bento_obj)

    def _get_latest(self, bento_name):
        with create_session(self.sess_maker) as sess:
            query = (
                sess.query(Bento)
                .filter_by(name=bento_name, deleted=False)
                .order_by(desc(Bento.created_at))
                .limit(1)
            )

            query_result = query.all()
            if len(query_result) == 1:
                return _bento_orm_obj_to_pb(query_result[0])
            else:
                return None

    def get(self, bento_name, bento_version="latest"):
        if bento_version.lower() == "latest":
            return self._get_latest(bento_name)

        with create_session(self.sess_maker) as sess:
            try:
                bento_obj = (
                    sess.query(Bento)
                    .filter_by(name=bento_name, version=bento_version)
                    .one()
                )
                if bento_obj.deleted:
                    # bento has been marked as deleted
                    return None
                return _bento_orm_obj_to_pb(bento_obj)
            except NoResultFound:
                return None

    def update_bento_service_metadata(
        self, bento_name, bento_version, bento_service_metadata_pb
    ):
        with create_session(self.sess_maker) as sess:
            try:
                bento_obj = (
                    sess.query(Bento)
                    .filter_by(name=bento_name, version=bento_version, deleted=False)
                    .one()
                )
                bento_obj.bento_service_metadata = ProtoMessageToDict(
                    bento_service_metadata_pb
                )
            except NoResultFound:
                raise YataiRepositoryException(
                    "Bento %s:%s is not found in repository" % bento_name, bento_version
                )

    def update_upload_status(self, bento_name, bento_version, upload_status_pb):
        with create_session(self.sess_maker) as sess:
            try:
                bento_obj = (
                    sess.query(Bento)
                    .filter_by(name=bento_name, version=bento_version, deleted=False)
                    .one()
                )
                # TODO:
                # if bento_obj.upload_status and bento_obj.upload_status.updated_at >
                # upload_status_pb.updated_at, update should be ignored
                bento_obj.upload_status = ProtoMessageToDict(upload_status_pb)
            except NoResultFound:
                raise YataiRepositoryException(
                    "Bento %s:%s is not found in repository" % bento_name, bento_version
                )

    def dangerously_delete(self, bento_name, bento_version):
        with create_session(self.sess_maker) as sess:
            try:
                bento_obj = (
                    sess.query(Bento)
                    .filter_by(name=bento_name, version=bento_version)
                    .one()
                )
                if bento_obj.deleted:
                    raise YataiRepositoryException(
                        "Bento {}:{} has already been deleted".format(
                            bento_name, bento_version
                        )
                    )
                bento_obj.deleted = True
            except NoResultFound:
                raise YataiRepositoryException(
                    "Bento %s:%s is not found in repository" % bento_name, bento_version
                )

    def list(
        self,
        bento_name=None,
        offset=None,
        limit=None,
        order_by=ListBentoRequest.created_at,
        ascending_order=False,
    ):
        with create_session(self.sess_maker) as sess:
            query = sess.query(Bento)
            order_by = ListBentoRequest.SORTABLE_COLUMN.Name(order_by)
            order_by_field = getattr(Bento, order_by)
            order_by_action = (
                order_by_field if ascending_order else desc(order_by_field)
            )
            query = query.order_by(order_by_action)
            if bento_name:
                # filter_by apply filtering criterion to a copy of the query
                query = query.filter_by(name=bento_name)
            query = query.filter_by(deleted=False)

            # We are not defaulting limit to 200 in the signature,
            # because protobuf will pass 0 as value
            limit = limit or 200
            # Limit and offset need to be called after order_by filter/filter_by is
            # called
            query = query.limit(limit)
            if offset:
                query = query.offset(offset)

            query_result = query.all()
            result = [_bento_orm_obj_to_pb(bento_obj) for bento_obj in query_result]
            return result
