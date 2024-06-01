from urllib.parse import urlparse

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured, PermissionDenied
from django.shortcuts import resolve_url

from gate.views import redirect_to_gate
from gate import REDIRECT_FIELD_NAME


class GateLockMixin:
    gate_url = None
    permission_denied_message = ''
    raise_exception = False
    redirect_field_name = REDIRECT_FIELD_NAME

    def get_gate_url(self):
        """
        Override this method to override the gate_url attribute.
        """
        gate_url = self.gate_url or settings.GATE_URL
        if not gate_url:
            raise ImproperlyConfigured(
                '{0} is missing the gate_url attribute. Define {0}.gate_url, settings.GATE_URL, or override '
                '{0}.get_gate_url().'.format(self.__class__.__name__)
            )
        return str(gate_url)

    def get_permission_denied_message(self):
        """
        Override this method to override the permission_denied_message attribute.
        """
        return self.permission_denied_message

    def get_redirect_field_name(self):
        """
        Override this method to override the redirect_field_name attribute.
        """
        return self.redirect_field_name

    def handle_no_permission(self):
        if self.raise_exception:
            raise PermissionDenied(self.get_permission_denied_message())

        path = self.request.build_absolute_uri()
        resolved_gate_url = resolve_url(self.get_gate_url())
        # If the gate url is the same scheme and net location then use the
        # path as the "next" url.
        gate_scheme, gate_netloc = urlparse(resolved_gate_url)[:2]
        current_scheme, current_netloc = urlparse(path)[:2]
        if (
            (not gate_scheme or gate_scheme == current_scheme) and
            (not gate_netloc or gate_netloc == current_netloc)
        ):
            path = self.request.get_full_path()
        return redirect_to_gate(
            path,
            resolved_gate_url,
            self.get_redirect_field_name(),
        )

    def lock_test_func(self, key):
        raise NotImplementedError(
            '{} is missing the implementation of the test_func() method.'.format(self.__class__.__name__)
        )

    def get_lock_test_func(self):
        """
        Override this method to use a different test_func method.
        """
        return self.lock_test_func

    def dispatch(self, request, *args, **kwargs):
        key = request.session.get('gate_key', None)
        key_test_result = self.get_lock_test_func()(key)
        if not key_test_result:
            return self.handle_no_permission()
        return super().dispatch(request, *args, **kwargs)
