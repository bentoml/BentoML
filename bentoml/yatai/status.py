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

from bentoml.yatai.proto.status_pb2 import Status as StatusProto


class Status(object):
    @staticmethod
    def Name(code):
        return StatusProto.Code.Name(code)

    @staticmethod
    def Value(status_name):
        return StatusProto.Code.Value(status_name)

    @staticmethod
    def OK(message=None):
        return StatusProto(status_code=StatusProto.OK, error_message=message)

    @staticmethod
    def CANCELLED(message=None):
        return StatusProto(status_code=StatusProto.CANCELLED, error_message=message)

    @staticmethod
    def UNKNOWN(message=None):
        return StatusProto(status_code=StatusProto.UNKNOWN, error_message=message)

    @staticmethod
    def INVALID_ARGUMENT(message=None):
        return StatusProto(
            status_code=StatusProto.INVALID_ARGUMENT, error_message=message
        )

    @staticmethod
    def DEADLINE_EXCEEDED(message=None):
        return StatusProto(
            status_code=StatusProto.DEADLINE_EXCEEDED, error_message=message
        )

    @staticmethod
    def NOT_FOUND(message=None):
        return StatusProto(status_code=StatusProto.NOT_FOUND, error_message=message)

    @staticmethod
    def ALREADY_EXISTS(message=None):
        return StatusProto(
            status_code=StatusProto.ALREADY_EXISTS, error_message=message
        )

    @staticmethod
    def PERMISSION_DENIED(message=None):
        return StatusProto(
            status_code=StatusProto.PERMISSION_DENIED, error_message=message
        )

    @staticmethod
    def UNAUTHENTICATED(message=None):
        return StatusProto(
            status_code=StatusProto.UNAUTHENTICATED, error_message=message
        )

    @staticmethod
    def RESOURCE_EXHAUSTED(message=None):
        return StatusProto(
            status_code=StatusProto.RESOURCE_EXHAUSTED, error_message=message
        )

    @staticmethod
    def FAILED_RECONDITION(message=None):
        return StatusProto(
            status_code=StatusProto.FAILED_PRECONDITION, error_message=message
        )

    @staticmethod
    def ABORTED(message=None):
        return StatusProto(status_code=StatusProto.ABORTED, error_message=message)

    @staticmethod
    def OUT_OF_RANGE(message=None):
        return StatusProto(status_code=StatusProto.OUT_OF_RANGE, error_message=message)

    @staticmethod
    def UNIMPLEMENTED(message=None):
        return StatusProto(status_code=StatusProto.UNIMPLEMENTED, error_message=message)

    @staticmethod
    def INTERNAL(message=None):
        return StatusProto(status_code=StatusProto.INTERNAL, error_message=message)

    @staticmethod
    def UNAVAILABLE(message=None):
        return StatusProto(status_code=StatusProto.UNAVAILABLE, error_message=message)

    @staticmethod
    def DATA_LOSS(message=None):
        return StatusProto(status_code=StatusProto.DATA_LOSS, error_message=message)
