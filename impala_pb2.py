# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: impala.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='impala.proto',
  package='message',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x0cimpala.proto\x12\x07message\"\'\n\x11TrajectoryRequest\x12\x12\n\ntrajectory\x18\x01 \x01(\t\"%\n\x12TrajectoryResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\"%\n\x10ParameterRequest\x12\x11\n\tparameter\x18\x01 \x01(\t\"$\n\x11ParameterResponse\x12\x0f\n\x07message\x18\x01 \x01(\x0c\x32\x9c\x01\n\x06IMPALA\x12I\n\x0eget_trajectory\x12\x1a.message.TrajectoryRequest\x1a\x1b.message.TrajectoryResponse\x12G\n\x0esend_parameter\x12\x19.message.ParameterRequest\x1a\x1a.message.ParameterResponseb\x06proto3'
)




_TRAJECTORYREQUEST = _descriptor.Descriptor(
  name='TrajectoryRequest',
  full_name='message.TrajectoryRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='trajectory', full_name='message.TrajectoryRequest.trajectory', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=25,
  serialized_end=64,
)


_TRAJECTORYRESPONSE = _descriptor.Descriptor(
  name='TrajectoryResponse',
  full_name='message.TrajectoryResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='message', full_name='message.TrajectoryResponse.message', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=66,
  serialized_end=103,
)


_PARAMETERREQUEST = _descriptor.Descriptor(
  name='ParameterRequest',
  full_name='message.ParameterRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='parameter', full_name='message.ParameterRequest.parameter', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=105,
  serialized_end=142,
)


_PARAMETERRESPONSE = _descriptor.Descriptor(
  name='ParameterResponse',
  full_name='message.ParameterResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='message', full_name='message.ParameterResponse.message', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
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
  serialized_start=144,
  serialized_end=180,
)

DESCRIPTOR.message_types_by_name['TrajectoryRequest'] = _TRAJECTORYREQUEST
DESCRIPTOR.message_types_by_name['TrajectoryResponse'] = _TRAJECTORYRESPONSE
DESCRIPTOR.message_types_by_name['ParameterRequest'] = _PARAMETERREQUEST
DESCRIPTOR.message_types_by_name['ParameterResponse'] = _PARAMETERRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TrajectoryRequest = _reflection.GeneratedProtocolMessageType('TrajectoryRequest', (_message.Message,), {
  'DESCRIPTOR' : _TRAJECTORYREQUEST,
  '__module__' : 'impala_pb2'
  # @@protoc_insertion_point(class_scope:message.TrajectoryRequest)
  })
_sym_db.RegisterMessage(TrajectoryRequest)

TrajectoryResponse = _reflection.GeneratedProtocolMessageType('TrajectoryResponse', (_message.Message,), {
  'DESCRIPTOR' : _TRAJECTORYRESPONSE,
  '__module__' : 'impala_pb2'
  # @@protoc_insertion_point(class_scope:message.TrajectoryResponse)
  })
_sym_db.RegisterMessage(TrajectoryResponse)

ParameterRequest = _reflection.GeneratedProtocolMessageType('ParameterRequest', (_message.Message,), {
  'DESCRIPTOR' : _PARAMETERREQUEST,
  '__module__' : 'impala_pb2'
  # @@protoc_insertion_point(class_scope:message.ParameterRequest)
  })
_sym_db.RegisterMessage(ParameterRequest)

ParameterResponse = _reflection.GeneratedProtocolMessageType('ParameterResponse', (_message.Message,), {
  'DESCRIPTOR' : _PARAMETERRESPONSE,
  '__module__' : 'impala_pb2'
  # @@protoc_insertion_point(class_scope:message.ParameterResponse)
  })
_sym_db.RegisterMessage(ParameterResponse)



_IMPALA = _descriptor.ServiceDescriptor(
  name='IMPALA',
  full_name='message.IMPALA',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=183,
  serialized_end=339,
  methods=[
  _descriptor.MethodDescriptor(
    name='get_trajectory',
    full_name='message.IMPALA.get_trajectory',
    index=0,
    containing_service=None,
    input_type=_TRAJECTORYREQUEST,
    output_type=_TRAJECTORYRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='send_parameter',
    full_name='message.IMPALA.send_parameter',
    index=1,
    containing_service=None,
    input_type=_PARAMETERREQUEST,
    output_type=_PARAMETERRESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_IMPALA)

DESCRIPTOR.services_by_name['IMPALA'] = _IMPALA

# @@protoc_insertion_point(module_scope)
