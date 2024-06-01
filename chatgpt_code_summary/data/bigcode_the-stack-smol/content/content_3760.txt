# Copyright (C) 2013 Google Inc., authors, and contributors <see AUTHORS file>
# Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>
# Created By: david@reciprocitylabs.com
# Maintained By: david@reciprocitylabs.com

from collections import namedtuple
from flask import session
from flask.ext.login import current_user
from .user_permissions import UserPermissions
from ggrc.models import get_model

Permission = namedtuple('Permission', 'action resource_type context_id')

_contributing_resource_types = {}

# Return a list of resource types using the same context space.
# This is needed because permissions may be given for, e.g., "Contract", but
#   the restriction on join is done knowing only "Directive".
def get_contributing_resource_types(resource_type):
  resource_types = _contributing_resource_types.get(resource_type, None)
  if resource_types is None:
    resource_types = [resource_type]
    resource_model = get_model(resource_type)
    if resource_model:
      resource_manager = resource_model._sa_class_manager
      resource_types.extend(
          manager.class_.__name__ for manager in
            resource_manager.subclass_managers(True))
    _contributing_resource_types[resource_type] = resource_types
  return resource_types

class DefaultUserPermissionsProvider(object):
  def __init__(self, settings):
    pass

  def permissions_for(self, user):
    return DefaultUserPermissions()

class DefaultUserPermissions(UserPermissions):
  # super user, context_id 0 indicates all contexts
  ADMIN_PERMISSION = Permission(
      '__GGRC_ADMIN__',
      '__GGRC_ALL__',
      0,
      )

  def _admin_permission_for_context(self, context_id):
    return Permission(
        self.ADMIN_PERMISSION.action,
        self.ADMIN_PERMISSION.resource_type,
        context_id)

  def _permission_match(self, permission, permissions):
    return permission.context_id in \
        permissions\
          .get(permission.action, {})\
          .get(permission.resource_type, [])

  def _is_allowed(self, permission):
    if 'permissions' not in session:
      return True
    permissions = session['permissions']
    if permissions is None:
      return True
    if self._permission_match(permission, permissions):
      return True
    if self._permission_match(self.ADMIN_PERMISSION, permissions):
      return True
    return self._permission_match(
        self._admin_permission_for_context(permission.context_id),
        permissions)

  def is_allowed_create(self, resource_type, context_id):
    """Whether or not the user is allowed to create a resource of the specified
    type in the context."""
    return self._is_allowed(Permission('create', resource_type, context_id))

  def is_allowed_read(self, resource_type, context_id):
    """Whether or not the user is allowed to read a resource of the specified
    type in the context."""
    return self._is_allowed(Permission('read', resource_type, context_id))

  def is_allowed_update(self, resource_type, context_id):
    """Whether or not the user is allowed to update a resource of the specified
    type in the context."""
    return self._is_allowed(Permission('update', resource_type, context_id))

  def is_allowed_delete(self, resource_type, context_id):
    """Whether or not the user is allowed to delete a resource of the specified
    type in the context."""
    return self._is_allowed(Permission('delete', resource_type, context_id))

  def _get_contexts_for(self, action, resource_type):
    # FIXME: (Security) When applicable, we should explicitly assert that no
    #   permissions are expected (e.g. that every user has ADMIN_PERMISSION).
    if 'permissions' not in session:
      return None
    permissions = session['permissions']
    if permissions is None:
      return None
    if self._permission_match(self.ADMIN_PERMISSION, permissions):
      return None

    # Get the list of contexts for a given resource type and any
    #   superclasses
    resource_types = get_contributing_resource_types(resource_type)

    ret = []
    for resource_type in resource_types:
      ret.extend(permissions.get(action, {}).get(resource_type, ()))

    # Extend with the list of all contexts for which the user is an ADMIN
    admin_list = list(
        permissions.get(self.ADMIN_PERMISSION.action, {})\
            .get(self.ADMIN_PERMISSION.resource_type, ()))
    ret.extend(admin_list)
    return ret

  def create_contexts_for(self, resource_type):
    """All contexts in which the user has create permission."""
    return self._get_contexts_for('create', resource_type)

  def read_contexts_for(self, resource_type):
    """All contexts in which the user has read permission."""
    return self._get_contexts_for('read', resource_type)

  def update_contexts_for(self, resource_type):
    """All contexts in which the user has update permission."""
    return self._get_contexts_for('update', resource_type)

  def delete_contexts_for(self, resource_type):
    """All contexts in which the user has delete permission."""
    return self._get_contexts_for('delete', resource_type)

