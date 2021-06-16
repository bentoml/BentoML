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
from bentoml.yatai.db import Base
from bentoml.yatai.db.stores.label import (
    LabelStore,
    RESOURCE_TYPE,
)
from bentoml.yatai.proto.repository_pb2 import (
    UploadStatus,
    BentoUri,
    BentoServiceMetadata,
    Bento as BentoPB,
    ListBentoRequest,
)

logger = logging.getLogger(__name__)

DEFAULT_UPLOAD_STATUS = UploadStatus(status=UploadStatus.UNINITIALIZED)
DEFAULT_LIST_LIMIT = 40


class Bento(Base):
    __tablename__ = 'bentos'
    __table_args__ = tuple(UniqueConstraint('name', 'version', name='_name_version_uc'))

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)

    # Storage URI for this Bento
    uri = Column(String, nullable=False)

    # Name is required for PostgreSQL and any future supported database which
    # requires an explicitly named type, or an explicitly named constraint in order to
    # generate the type and/or a table that uses it.
    uri_type = Column(
        Enum(*BentoUri.StorageType.keys(), name='uri_type'), default=BentoUri.UNSET
    )

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


def _bento_orm_obj_to_pb(bento_obj, labels=None):
    # Backwards compatible support loading saved bundle created before 0.8.0
    if (
        'apis' in bento_obj.bento_service_metadata
        and bento_obj.bento_service_metadata['apis']
    ):
        for api in bento_obj.bento_service_metadata['apis']:
            if 'handler_type' in api:
                api['input_type'] = api['handler_type']
                del api['handler_type']
            if 'handler_config' in api:
                api['input_config'] = api['handler_config']
                del api['handler_config']
            if 'output_type' not in api:
                api['output_type'] = 'DefaultOutput'

    bento_service_metadata_pb = ParseDict(
        bento_obj.bento_service_metadata, BentoServiceMetadata()
    )
    bento_uri = BentoUri(
        uri=bento_obj.uri, type=BentoUri.StorageType.Value(bento_obj.uri_type)
    )
    if not bento_obj.upload_status:
        upload_status = DEFAULT_UPLOAD_STATUS
    else:
        upload_status = UploadStatus(
            status=UploadStatus.Status.Value(bento_obj.upload_status['status'])
        )
    if labels is not None:
        bento_service_metadata_pb.labels.update(labels)
    return BentoPB(
        name=bento_obj.name,
        version=bento_obj.version,
        uri=bento_uri,
        bento_service_metadata=bento_service_metadata_pb,
        status=upload_status,
    )


class MetadataStore(object):
    @staticmethod
    def add(sess, bento_name, bento_version, uri, uri_type):
        bento_obj = Bento()
        bento_obj.name = bento_name
        bento_obj.version = bento_version
        bento_obj.uri = uri
        bento_obj.uri_type = BentoUri.StorageType.Name(uri_type)
        return sess.add(bento_obj)

    @staticmethod
    def _get_latest(sess, bento_name):
        query = (
            sess.query(Bento)
            .filter_by(name=bento_name, deleted=False)
            .order_by(desc(Bento.created_at))
            .limit(1)
        )

        query_result = query.all()
        if len(query_result) == 1:
            labels = LabelStore.get(sess, RESOURCE_TYPE.bento, query_result[0].id)
            return _bento_orm_obj_to_pb(query_result[0], labels)
        else:
            return None

    @staticmethod
    def get(sess, bento_name, bento_version="latest"):
        if bento_version.lower() == "latest":
            return MetadataStore._get_latest(sess, bento_name)

        try:
            bento_obj = (
                sess.query(Bento)
                .filter_by(name=bento_name, version=bento_version, deleted=False)
                .one()
            )
            if bento_obj.deleted:
                # bento has been marked as deleted
                return None
            labels = LabelStore.get(sess, RESOURCE_TYPE.bento, bento_obj.id)
            return _bento_orm_obj_to_pb(bento_obj, labels)
        except NoResultFound:
            return None

    @staticmethod
    def update(sess, bento_name, bento_version, bento_service_metadata_pb):
        try:
            bento_obj = (
                sess.query(Bento)
                .filter_by(name=bento_name, version=bento_version, deleted=False)
                .one()
            )
            service_metadata = ProtoMessageToDict(bento_service_metadata_pb)
            bento_obj.bento_service_metadata = service_metadata
            if service_metadata.get('labels', None) is not None:
                bento = (
                    sess.query(Bento)
                    .filter_by(name=bento_name, version=bento_version)
                    .one()
                )
                LabelStore.add_or_update(
                    sess, RESOURCE_TYPE.bento, bento.id, service_metadata['labels']
                )
        except NoResultFound:
            raise YataiRepositoryException(
                "Bento %s:%s is not found in repository" % bento_name, bento_version
            )

    @staticmethod
    def update_upload_status(sess, bento_name, bento_version, upload_status_pb):
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

    @staticmethod
    def dangerously_delete(sess, bento_name, bento_version):
        try:
            bento_obj = (
                sess.query(Bento)
                .filter_by(name=bento_name, version=bento_version, deleted=False)
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

    @staticmethod
    def list(
        sess,
        bento_name=None,
        offset=None,
        limit=None,
        label_selectors=None,
        order_by=ListBentoRequest.created_at,
        ascending_order=False,
    ):
        query = sess.query(Bento)
        order_by = ListBentoRequest.SORTABLE_COLUMN.Name(order_by)
        order_by_field = getattr(Bento, order_by)
        order_by_action = order_by_field if ascending_order else desc(order_by_field)
        query = query.order_by(order_by_action)
        if bento_name:
            # filter_by apply filtering criterion to a copy of the query
            query = query.filter_by(name=bento_name)
        query = query.filter_by(deleted=False)
        if label_selectors is not None and (
            label_selectors.match_labels or label_selectors.match_expressions
        ):
            bento_ids = LabelStore.filter_query(
                sess, RESOURCE_TYPE.bento, label_selectors
            )
            query = query.filter(Bento.id.in_(bento_ids))

        # We are not defaulting limit to 200 in the signature,
        # because protobuf will pass 0 as value
        limit = limit or DEFAULT_LIST_LIMIT
        # Limit and offset need to be called after order_by filter/filter_by is
        # called
        query = query.limit(limit)
        if offset:
            query = query.offset(offset)

        query_result = query.all()
        bento_ids = [bento_obj.id for bento_obj in query_result]
        labels = LabelStore.list(sess, RESOURCE_TYPE.bento, bento_ids)
        result = [
            _bento_orm_obj_to_pb(bento_obj, labels.get(str(bento_obj.id)))
            for bento_obj in query_result
        ]
        return result
