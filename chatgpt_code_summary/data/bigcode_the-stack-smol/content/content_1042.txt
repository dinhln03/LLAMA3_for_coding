"""Permissions for the client app"""
from rest_framework import permissions


class ClientPermissions(permissions.BasePermission):
    """Handles authorization of requests to the client app."""
    def has_permission(self, request, view):
        if view.action == 'create' \
                and request.user.has_perm('client.add_client'):
            return True

        if view.action in ['update', 'partial_update'] \
                and request.user.has_perm('client.change_client'):
            return True

        if view.action in ['list', 'retrieve'] \
                and request.user.has_perm('client.view_client'):
            return True
