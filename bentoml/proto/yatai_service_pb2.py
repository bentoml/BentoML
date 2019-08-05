# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: yatai_service.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
import bentoml.proto.status_pb2 as status__pb2
import deployment_pb2 as deployment__pb2
import repository_pb2 as repository__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='yatai_service.proto',
  package='bentoml',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x13yatai_service.proto\x12\x07\x62\x65ntoml\x1a\x1bgoogle/protobuf/empty.proto\x1a\x0cstatus.proto\x1a\x10\x64\x65ployment.proto\x1a\x10repository.proto\"6\n\x13HealthCheckResponse\x12\x1f\n\x06status\x18\x01 \x01(\x0b\x32\x0f.bentoml.Status\"1\n\x1eGetYataiServiceVersionResponse\x12\x0f\n\x07version\x18\x01 \x01(\t\"\x18\n\x05\x43hunk\x12\x0f\n\x07\x63ontent\x18\x01 \x01(\x0c\x32\xe3\x06\n\x05Yatai\x12\x43\n\x0bHealthCheck\x12\x16.google.protobuf.Empty\x1a\x1c.bentoml.HealthCheckResponse\x12Y\n\x16GetYataiServiceVersion\x12\x16.google.protobuf.Empty\x1a\'.bentoml.GetYataiServiceVersionResponse\x12T\n\x0f\x41pplyDeployment\x12\x1f.bentoml.ApplyDeploymentRequest\x1a .bentoml.ApplyDeploymentResponse\x12W\n\x10\x44\x65leteDeployment\x12 .bentoml.DeleteDeploymentRequest\x1a!.bentoml.DeleteDeploymentResponse\x12N\n\rGetDeployment\x12\x1d.bentoml.GetDeploymentRequest\x1a\x1e.bentoml.GetDeploymentResponse\x12]\n\x12\x44\x65scribeDeployment\x12\".bentoml.DescribeDeploymentRequest\x1a#.bentoml.DescribeDeploymentResponse\x12T\n\x0fListDeployments\x12\x1f.bentoml.ListDeploymentsRequest\x1a .bentoml.ListDeploymentsResponse\x12\x37\n\x08\x41\x64\x64\x42\x65nto\x12\x0e.bentoml.Chunk\x1a\x19.bentoml.AddBentoResponse(\x01\x12H\n\x0bRemoveBento\x12\x1b.bentoml.RemoveBentoRequest\x1a\x1c.bentoml.RemoveBentoResponse\x12?\n\x08GetBento\x12\x18.bentoml.GetBentoRequest\x1a\x19.bentoml.GetBentoResponse\x12\x42\n\tListBento\x12\x19.bentoml.ListBentoRequest\x1a\x1a.bentoml.ListBentoResponseb\x06proto3')
  ,
  dependencies=[google_dot_protobuf_dot_empty__pb2.DESCRIPTOR,status__pb2.DESCRIPTOR,deployment__pb2.DESCRIPTOR,repository__pb2.DESCRIPTOR,])




_HEALTHCHECKRESPONSE = _descriptor.Descriptor(
  name='HealthCheckResponse',
  full_name='bentoml.HealthCheckResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='bentoml.HealthCheckResponse.status', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=111,
  serialized_end=165,
)


_GETYATAISERVICEVERSIONRESPONSE = _descriptor.Descriptor(
  name='GetYataiServiceVersionResponse',
  full_name='bentoml.GetYataiServiceVersionResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='version', full_name='bentoml.GetYataiServiceVersionResponse.version', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=167,
  serialized_end=216,
)


_CHUNK = _descriptor.Descriptor(
  name='Chunk',
  full_name='bentoml.Chunk',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='content', full_name='bentoml.Chunk.content', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=218,
  serialized_end=242,
)

_HEALTHCHECKRESPONSE.fields_by_name['status'].message_type = status__pb2._STATUS
DESCRIPTOR.message_types_by_name['HealthCheckResponse'] = _HEALTHCHECKRESPONSE
DESCRIPTOR.message_types_by_name['GetYataiServiceVersionResponse'] = _GETYATAISERVICEVERSIONRESPONSE
DESCRIPTOR.message_types_by_name['Chunk'] = _CHUNK
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

HealthCheckResponse = _reflection.GeneratedProtocolMessageType('HealthCheckResponse', (_message.Message,), {
  'DESCRIPTOR' : _HEALTHCHECKRESPONSE,
  '__module__' : 'yatai_service_pb2'
  # @@protoc_insertion_point(class_scope:bentoml.HealthCheckResponse)
  })
_sym_db.RegisterMessage(HealthCheckResponse)

GetYataiServiceVersionResponse = _reflection.GeneratedProtocolMessageType('GetYataiServiceVersionResponse', (_message.Message,), {
  'DESCRIPTOR' : _GETYATAISERVICEVERSIONRESPONSE,
  '__module__' : 'yatai_service_pb2'
  # @@protoc_insertion_point(class_scope:bentoml.GetYataiServiceVersionResponse)
  })
_sym_db.RegisterMessage(GetYataiServiceVersionResponse)

Chunk = _reflection.GeneratedProtocolMessageType('Chunk', (_message.Message,), {
  'DESCRIPTOR' : _CHUNK,
  '__module__' : 'yatai_service_pb2'
  # @@protoc_insertion_point(class_scope:bentoml.Chunk)
  })
_sym_db.RegisterMessage(Chunk)



_YATAI = _descriptor.ServiceDescriptor(
  name='Yatai',
  full_name='bentoml.Yatai',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=245,
  serialized_end=1112,
  methods=[
  _descriptor.MethodDescriptor(
    name='HealthCheck',
    full_name='bentoml.Yatai.HealthCheck',
    index=0,
    containing_service=None,
    input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    output_type=_HEALTHCHECKRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetYataiServiceVersion',
    full_name='bentoml.Yatai.GetYataiServiceVersion',
    index=1,
    containing_service=None,
    input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    output_type=_GETYATAISERVICEVERSIONRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='ApplyDeployment',
    full_name='bentoml.Yatai.ApplyDeployment',
    index=2,
    containing_service=None,
    input_type=deployment__pb2._APPLYDEPLOYMENTREQUEST,
    output_type=deployment__pb2._APPLYDEPLOYMENTRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='DeleteDeployment',
    full_name='bentoml.Yatai.DeleteDeployment',
    index=3,
    containing_service=None,
    input_type=deployment__pb2._DELETEDEPLOYMENTREQUEST,
    output_type=deployment__pb2._DELETEDEPLOYMENTRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetDeployment',
    full_name='bentoml.Yatai.GetDeployment',
    index=4,
    containing_service=None,
    input_type=deployment__pb2._GETDEPLOYMENTREQUEST,
    output_type=deployment__pb2._GETDEPLOYMENTRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='DescribeDeployment',
    full_name='bentoml.Yatai.DescribeDeployment',
    index=5,
    containing_service=None,
    input_type=deployment__pb2._DESCRIBEDEPLOYMENTREQUEST,
    output_type=deployment__pb2._DESCRIBEDEPLOYMENTRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='ListDeployments',
    full_name='bentoml.Yatai.ListDeployments',
    index=6,
    containing_service=None,
    input_type=deployment__pb2._LISTDEPLOYMENTSREQUEST,
    output_type=deployment__pb2._LISTDEPLOYMENTSRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='AddBento',
    full_name='bentoml.Yatai.AddBento',
    index=7,
    containing_service=None,
    input_type=_CHUNK,
    output_type=repository__pb2._ADDBENTORESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='RemoveBento',
    full_name='bentoml.Yatai.RemoveBento',
    index=8,
    containing_service=None,
    input_type=repository__pb2._REMOVEBENTOREQUEST,
    output_type=repository__pb2._REMOVEBENTORESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetBento',
    full_name='bentoml.Yatai.GetBento',
    index=9,
    containing_service=None,
    input_type=repository__pb2._GETBENTOREQUEST,
    output_type=repository__pb2._GETBENTORESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='ListBento',
    full_name='bentoml.Yatai.ListBento',
    index=10,
    containing_service=None,
    input_type=repository__pb2._LISTBENTOREQUEST,
    output_type=repository__pb2._LISTBENTORESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_YATAI)

DESCRIPTOR.services_by_name['Yatai'] = _YATAI

# @@protoc_insertion_point(module_scope)
