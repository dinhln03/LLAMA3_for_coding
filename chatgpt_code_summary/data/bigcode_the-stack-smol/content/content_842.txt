from importlib import import_module

from rest_framework import status
from rest_framework.response import Response
from rest_framework.viewsets import GenericViewSet


class HookViewSet(GenericViewSet):

    def post(self, request, *args, **kwargs):
        data = request.data
        action = data['action']
        event = request.META.get('HTTP_X_GITHUB_EVENT', None)
        if not event:
            return Response({'result': False}, status=status.HTTP_200_OK)
        if 'installation' in event:
            event = 'installation'
        try:
            dirname = __name__.split('viewsets')[0]
            module = import_module(f'{dirname}{event}.api')
            result = getattr(module, f'hook_{action}')(data)
        except ImportError:
            result = False
        return Response({'result': result}, status.HTTP_200_OK)
