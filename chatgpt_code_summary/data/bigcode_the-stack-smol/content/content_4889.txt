"""
Based on REST Framework Parsers, optimized for csv

Parsers are used to parse the content of incoming HTTP requests.

They give us a generic way of being able to handle various media types
on the request, such as form content or json encoded data.
"""
import codecs
from urllib import parse

from django.conf import settings
from django.core.files.uploadhandler import StopFutureHandlers
from django.http import QueryDict
from django.http.multipartparser import ChunkIter
from django.http.multipartparser import \
    MultiPartParser as DjangoMultiPartParser
from django.http.multipartparser import MultiPartParserError, parse_header
from django.utils.encoding import force_str

from rest_framework import renderers
from rest_framework.exceptions import ParseError
from rest_framework.settings import api_settings
from rest_framework.utils import json


class DataAndFiles:
    def __init__(self, data, files):
        self.data = data
        self.files = files


class BaseParser:
    """
    All parsers should extend `BaseParser`, specifying a `media_type`
    attribute, and overriding the `.parse()` method.
    """
    media_type = None

    def parse(self, stream, media_type=None, parser_context=None):
        """
        Given a stream to read from, return the parsed representation.
        Should return parsed data, or a `DataAndFiles` object consisting of the
        parsed data and files.
        """
        raise NotImplementedError(".parse() must be overridden.")

class MParser(BaseParser):
    """
    Parser for file upload data.
    """
    media_type = '*/*'
    errors = {
        'unhandled': 'FileUpload parse error - none of upload handlers can handle the stream',
        'no_filename': 'Missing filename. Request should include a Content-Disposition header with a filename parameter.',
    }

    def parse(self, stream, media_type=None, parser_context=None):
        """
        Treats the incoming bytestream as a raw file upload and returns
        a `DataAndFiles` object.

        `.data` will be None (we expect request body to be a file content).
        `.files` will be a `QueryDict` containing one 'file' element.
        """
        parser_context = parser_context or {}
        request = parser_context['request']
        encoding = parser_context.get('encoding', settings.DEFAULT_CHARSET)
        meta = request.META
        upload_handlers = request.upload_handlers
        filename = self.get_filename(stream, media_type, parser_context)

        # Note that this code is extracted from Django's handling of
        # file uploads in MultiPartParser.
        content_type = meta.get('HTTP_CONTENT_TYPE',
                                meta.get('CONTENT_TYPE', ''))
        try:
            content_length = int(meta.get('HTTP_CONTENT_LENGTH',
                                          meta.get('CONTENT_LENGTH', 0)))
        except (ValueError, TypeError):
            content_length = None

        # See if the handler will want to take care of the parsing.
        for handler in upload_handlers:
            result = handler.handle_raw_input(stream,
                                              meta,
                                              content_length,
                                              None,
                                              encoding)
            if result is not None:
                return DataAndFiles({}, {'file': result[1]})

        # This is the standard case.
        possible_sizes = [x.chunk_size for x in upload_handlers if x.chunk_size]
        chunk_size = min([2 ** 31 - 4] + possible_sizes)
        chunks = ChunkIter(stream, chunk_size)
        counters = [0] * len(upload_handlers)

        for index, handler in enumerate(upload_handlers):
            try:
                handler.new_file(None, filename, content_type,
                                 content_length, encoding)
            except StopFutureHandlers:
                upload_handlers = upload_handlers[:index + 1]
                break

        for chunk in chunks:
            for index, handler in enumerate(upload_handlers):
                """
                Trimming HttpResponse encapsulation from parsed file stream
                """
                chunk_length = len(chunk)
                start = chunk.find(bytes('\n\r\n','utf-8')) + 3
                end = chunk.rfind(bytes('\r\n','utf-8'))
                end = chunk[:end].rfind(bytes('\r\n','utf-8')) + 2
                chunk = handler.receive_data_chunk(chunk[start:end], counters[index])
                counters[index] += chunk_length
                if chunk is None:
                    break

        for index, handler in enumerate(upload_handlers):
            file_obj = handler.file_complete(counters[index])
            if file_obj is not None:
                return DataAndFiles({}, {'file': file_obj})

        raise ParseError(self.errors['unhandled'])

    def get_filename(self, stream, media_type, parser_context):
        """
        Detects the uploaded file name. First searches a 'filename' url kwarg.
        Then tries to parse Content-Disposition header.
        """
        try:
            return parser_context['kwargs']['filename']
        except KeyError:
            pass

        try:
            meta = parser_context['request'].META
            disposition = parse_header(meta['HTTP_CONTENT_DISPOSITION'].encode())
            filename_parm = disposition[1]
            if 'filename*' in filename_parm:
                return self.get_encoded_filename(filename_parm)
            return force_str(filename_parm['filename'])
        except (AttributeError, KeyError, ValueError):
            pass

    def get_encoded_filename(self, filename_parm):
        """
        Handle encoded filenames per RFC6266. See also:
        https://tools.ietf.org/html/rfc2231#section-4
        """
        encoded_filename = force_str(filename_parm['filename*'])
        try:
            charset, lang, filename = encoded_filename.split('\'', 2)
            filename = parse.unquote(filename)
        except (ValueError, LookupError):
            filename = force_str(filename_parm['filename'])
        return filename
