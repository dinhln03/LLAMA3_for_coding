#coding=utf-8
#
# Copyright (C) 2015 Feigr TECH Co., Ltd. All rights reserved.
# Created on 2013-8-13, by Junn
#
#
#import settings

from django.middleware.csrf import get_token
from django.http.response import Http404
from django.core.exceptions import PermissionDenied

from rest_framework.generics import GenericAPIView
from rest_framework import exceptions, status
from rest_framework.response import Response

from core.authentication import CsrfError
from utils.http import JResponse
from core import codes


def csrf_failure(request, reason=''):
    """
    customize the response for csrf_token invalid
    """
    # if request.is_ajax():
    #     return JResponse(codes.get('csrf_invalid'))
    # return
    get_token(request)
    return JResponse(codes.get('csrf_invalid'), status=403)


class CustomAPIView(GenericAPIView):
    """
    customize the APIView for customize exception response
    """
    
    def handle_exception(self, exc):
        """
        Handle any exception that occurs, by returning an appropriate response,
        or re-raising the error.
        """
        if isinstance(exc, exceptions.Throttled):
            # Throttle wait header
            self.headers['X-Throttle-Wait-Seconds'] = '%d' % exc.wait

        if isinstance(exc, (exceptions.NotAuthenticated,
                            exceptions.AuthenticationFailed)):
            # WWW-Authenticate header for 401 responses, else coerce to 403
            auth_header = self.get_authenticate_header(self.request)

            if auth_header:
                self.headers['WWW-Authenticate'] = auth_header
            else:
                exc.status_code = status.HTTP_403_FORBIDDEN

        if isinstance(exc, exceptions.MethodNotAllowed):
            return Response(codes.get('invalid_request_method'),
                            status=exc.status_code,
                            exception=True)

        elif isinstance(exc, CsrfError):
            return Response(codes.get('csrf_invalid'),
                            status=exc.status_code,
                            exception=True)

        elif isinstance(exc, exceptions.ParseError):
            return Response(codes.get('parse_error'),
                            status=exc.status_code,
                            exception=True)

        elif isinstance(exc, exceptions.AuthenticationFailed):
            return Response(codes.get('authentication_failed'),
                            status=exc.status_code,
                            exception=True)

        elif isinstance(exc, exceptions.NotAuthenticated):
            return Response(codes.get('not_authenticated'),
                            status=exc.status_code,
                            exception=True)

        elif isinstance(exc, exceptions.PermissionDenied):
            return Response(codes.get('permission_denied'),
                            status=exc.status_code,
                            exception=True)

        elif isinstance(exc, exceptions.NotAcceptable):
            return Response(codes.get('not_acceptable'),
                            status=exc.status_code,
                            exception=True)

        elif isinstance(exc, exceptions.UnsupportedMediaType):
            return Response(codes.get('unsupported_media_type'),
                            status=exc.status_code,
                            exception=True)

        elif isinstance(exc, exceptions.Throttled):
            return Response(codes.get('throttled'),
                            status=exc.status_code,
                            exception=True)

        elif isinstance(exc, Http404):
            return Response(codes.get('not_found'),
                            status=status.HTTP_404_NOT_FOUND,
                            exception=True)

        elif isinstance(exc, PermissionDenied):
            return Response(codes.get('permission_denied'),
                            status=status.HTTP_403_FORBIDDEN,
                            exception=True)
        raise
