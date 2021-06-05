import os
import io
import tarfile
from bentoml.yatai.proto.repository_pb2 import UploadBentoRequest, DownloadBentoResponse


DEFAULT_FILE_CHUNK_SIZE = 1024 * 8


class FileStream:
    def __init__(self):
        self.buffer = io.BytesIO()
        self.offset = 0

    def write(self, s):
        self.buffer.write(s)
        self.offset += len(s)

    def tell(self):
        return self.offset

    def close(self):
        self.buffer.close()

    def read_value(self):
        try:
            return self.buffer.getvalue()
        finally:
            self.buffer.close()
            self.buffer = io.BytesIO()


class BentoBundleStreamRequestsOrResponses:
    def __init__(
        self,
        bento_name,
        bento_version,
        directory_path,
        file_chunk_size=DEFAULT_FILE_CHUNK_SIZE,
        is_request=True,
    ):
        """
        Build a tar file chunk by chunk on the fly from a directory and then stream out
        either request or response for upload/download.

        Args:
            bento_name:
            bento_version:
            directory_path:
            file_chunk_size:
            is_request:
        """
        self.bento_name = bento_name
        self.bento_version = bento_version
        self.directory_path = directory_path
        self.file_chunk_size = file_chunk_size
        self.is_request = is_request

    @staticmethod
    def _stream_file_into_tar(tarinfo, tar, file_handler, buf_size):
        out = tar.fileobj
        """
        For iter(), the second argument, sentinel, is given, then object must
        be a callable object. The iterator created in this case will
        call object with no arguments for each call to its __next__()
        method; if the value returned is equal to sentinel, StopIteration
        will be raised, otherwise the value will be returned.
        """
        for b in iter(lambda: file_handler.read(buf_size), b''):
            out.write(b)
            yield
        blocks, remainder = divmod(tarinfo.size, tarfile.BLOCKSIZE)
        if remainder > 0:
            out.write(tarfile.NUL * (tarfile.BLOCKSIZE - remainder))
            blocks += 1
        tar.offset += blocks * tarfile.BLOCKSIZE
        yield

    def create_request_or_response(self, value):
        if self.is_request:
            return UploadBentoRequest(
                bento_name=self.bento_name,
                bento_version=self.bento_version,
                bento_bundle=value,
            )
        else:
            return DownloadBentoResponse(bento_bundle=value)

    def __iter__(self):
        if self.is_request:
            yield UploadBentoRequest(
                bento_name=self.bento_name, bento_version=self.bento_version
            )

        out_stream = FileStream()
        tar = tarfile.TarFile(fileobj=out_stream, mode='w')
        # Include length of separator.
        prefix_len = len(self.directory_path) + len(os.path.sep)

        # Add the directory path to tar
        tar.add(name=self.directory_path, arcname='', recursive=False)

        # Manually walk the directory and add to tarfile
        for path, dirs, files in os.walk(self.directory_path):
            arc_path = path[prefix_len:]

            # Add files to tar. Add the tar info and then stream the file into tar
            for f in files:
                file_path = os.path.join(path, f)
                with open(file_path, 'rb') as file:
                    # Reading the file info and generate tarinfo from that.
                    # No file content is added to the tar at this point
                    tar_info = tar.gettarinfo(
                        name=file_path, arcname=os.path.join(arc_path, f), fileobj=file,
                    )
                    tar.addfile(tar_info)
                    # stream the file content into the tar
                    for _ in self._stream_file_into_tar(
                        tar_info, tar, file, self.file_chunk_size
                    ):
                        # Stream out
                        yield self.create_request_or_response(out_stream.read_value())

            # Add the directory path to tar
            for dir_path in dirs:
                tar.add(
                    name=os.path.join(path, dir_path),
                    arcname=os.path.join(arc_path, dir_path),
                    recursive=False,
                )
            # Stream out directories info in the tar
            yield self.create_request_or_response(out_stream.read_value())
        tar.close()
        yield self.create_request_or_response(out_stream.read_value())
        out_stream.close()
