# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: label_selectors.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name='label_selectors.proto',
    package='bentoml',
    syntax='proto3',
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\x15label_selectors.proto\x12\x07\x62\x65ntoml\"\x9c\x03\n\x0eLabelSelectors\x12>\n\x0cmatch_labels\x18\x01 \x03(\x0b\x32(.bentoml.LabelSelectors.MatchLabelsEntry\x12J\n\x11match_expressions\x18\x02 \x03(\x0b\x32/.bentoml.LabelSelectors.LabelSelectorExpression\x1a\xc9\x01\n\x17LabelSelectorExpression\x12O\n\x08operator\x18\x01 \x01(\x0e\x32=.bentoml.LabelSelectors.LabelSelectorExpression.OPERATOR_TYPE\x12\x0b\n\x03key\x18\x02 \x01(\t\x12\x0e\n\x06values\x18\x03 \x03(\t\"@\n\rOPERATOR_TYPE\x12\x06\n\x02In\x10\x00\x12\t\n\x05NotIn\x10\x01\x12\n\n\x06\x45xists\x10\x02\x12\x10\n\x0c\x44oesNotExist\x10\x03\x1a\x32\n\x10MatchLabelsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x62\x06proto3',
)


_LABELSELECTORS_LABELSELECTOREXPRESSION_OPERATOR_TYPE = _descriptor.EnumDescriptor(
    name='OPERATOR_TYPE',
    full_name='bentoml.LabelSelectors.LabelSelectorExpression.OPERATOR_TYPE',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
        _descriptor.EnumValueDescriptor(
            name='In',
            index=0,
            number=0,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='NotIn',
            index=1,
            number=1,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='Exists',
            index=2,
            number=2,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='DoesNotExist',
            index=3,
            number=3,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    containing_type=None,
    serialized_options=None,
    serialized_start=331,
    serialized_end=395,
)
_sym_db.RegisterEnumDescriptor(_LABELSELECTORS_LABELSELECTOREXPRESSION_OPERATOR_TYPE)


_LABELSELECTORS_LABELSELECTOREXPRESSION = _descriptor.Descriptor(
    name='LabelSelectorExpression',
    full_name='bentoml.LabelSelectors.LabelSelectorExpression',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='operator',
            full_name='bentoml.LabelSelectors.LabelSelectorExpression.operator',
            index=0,
            number=1,
            type=14,
            cpp_type=8,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name='key',
            full_name='bentoml.LabelSelectors.LabelSelectorExpression.key',
            index=1,
            number=2,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name='values',
            full_name='bentoml.LabelSelectors.LabelSelectorExpression.values',
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[_LABELSELECTORS_LABELSELECTOREXPRESSION_OPERATOR_TYPE,],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=194,
    serialized_end=395,
)

_LABELSELECTORS_MATCHLABELSENTRY = _descriptor.Descriptor(
    name='MatchLabelsEntry',
    full_name='bentoml.LabelSelectors.MatchLabelsEntry',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='key',
            full_name='bentoml.LabelSelectors.MatchLabelsEntry.key',
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name='value',
            full_name='bentoml.LabelSelectors.MatchLabelsEntry.value',
            index=1,
            number=2,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=b'8\001',
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=397,
    serialized_end=447,
)

_LABELSELECTORS = _descriptor.Descriptor(
    name='LabelSelectors',
    full_name='bentoml.LabelSelectors',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='match_labels',
            full_name='bentoml.LabelSelectors.match_labels',
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name='match_expressions',
            full_name='bentoml.LabelSelectors.match_expressions',
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[
        _LABELSELECTORS_LABELSELECTOREXPRESSION,
        _LABELSELECTORS_MATCHLABELSENTRY,
    ],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=35,
    serialized_end=447,
)

_LABELSELECTORS_LABELSELECTOREXPRESSION.fields_by_name[
    'operator'
].enum_type = _LABELSELECTORS_LABELSELECTOREXPRESSION_OPERATOR_TYPE
_LABELSELECTORS_LABELSELECTOREXPRESSION.containing_type = _LABELSELECTORS
_LABELSELECTORS_LABELSELECTOREXPRESSION_OPERATOR_TYPE.containing_type = (
    _LABELSELECTORS_LABELSELECTOREXPRESSION
)
_LABELSELECTORS_MATCHLABELSENTRY.containing_type = _LABELSELECTORS
_LABELSELECTORS.fields_by_name[
    'match_labels'
].message_type = _LABELSELECTORS_MATCHLABELSENTRY
_LABELSELECTORS.fields_by_name[
    'match_expressions'
].message_type = _LABELSELECTORS_LABELSELECTOREXPRESSION
DESCRIPTOR.message_types_by_name['LabelSelectors'] = _LABELSELECTORS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

LabelSelectors = _reflection.GeneratedProtocolMessageType(
    'LabelSelectors',
    (_message.Message,),
    {
        'LabelSelectorExpression': _reflection.GeneratedProtocolMessageType(
            'LabelSelectorExpression',
            (_message.Message,),
            {
                'DESCRIPTOR': _LABELSELECTORS_LABELSELECTOREXPRESSION,
                '__module__': 'label_selectors_pb2'
                # @@protoc_insertion_point(class_scope:bentoml.LabelSelectors.LabelSelectorExpression)
            },
        ),
        'MatchLabelsEntry': _reflection.GeneratedProtocolMessageType(
            'MatchLabelsEntry',
            (_message.Message,),
            {
                'DESCRIPTOR': _LABELSELECTORS_MATCHLABELSENTRY,
                '__module__': 'label_selectors_pb2'
                # @@protoc_insertion_point(class_scope:bentoml.LabelSelectors.MatchLabelsEntry)
            },
        ),
        'DESCRIPTOR': _LABELSELECTORS,
        '__module__': 'label_selectors_pb2'
        # @@protoc_insertion_point(class_scope:bentoml.LabelSelectors)
    },
)
_sym_db.RegisterMessage(LabelSelectors)
_sym_db.RegisterMessage(LabelSelectors.LabelSelectorExpression)
_sym_db.RegisterMessage(LabelSelectors.MatchLabelsEntry)


_LABELSELECTORS_MATCHLABELSENTRY._options = None
# @@protoc_insertion_point(module_scope)
