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
)
from sqlalchemy.orm.exc import NoResultFound
from google.protobuf.json_format import ParseDict

from bentoml.utils import ProtoMessageToDict
from bentoml.exceptions import BentoMLRepositoryException
from bentoml.db import Base, create_session
from bentoml.proto.repository_pb2 import (
    UploadStatus,
    BentoUri,
    BentoServiceMetadata,
    Bento as BentoPB,
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

    def get(self, bento_name, bento_version):
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
                raise BentoMLRepositoryException(
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
                raise BentoMLRepositoryException(
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
                if not bento_obj.deleted:
                    raise BentoMLRepositoryException(
                        "Bento %s:%s has already been deleted" % bento_name,
                        bento_version,
                    )
                bento_obj.deleted = True
            except NoResultFound:
                raise BentoMLRepositoryException(
                    "Bento %s:%s is not found in repository" % bento_name, bento_version
                )

    def list(self, bento_name=None, offset=None, limit=None, filter_str=None):
        with create_session(self.sess_maker) as sess:
            query = sess.query(Bento)
            if limit:
                query.limit(limit)
            if offset:
                query.offset(offset)
            if filter_str:
                query.filter(Bento.name.contains(filter_str))
            if bento_name:
                query.filter_by(name=bento_name)

            return query.all()
