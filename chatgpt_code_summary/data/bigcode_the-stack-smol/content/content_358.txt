import datetime

from . import status
from .errors import InvalidAuthRequest, ProtocolVersionUnsupported, NoMutualAuthType
from .signing import Key
from .response import AuthResponse

class AuthPrincipal:
    def __init__(self, userid, auth_methods, ptags=None, session_expiry=None):
        self.userid = userid
        self.auth_methods = auth_methods
        if ptags is None:
            ptags = []
        self.ptags = ptags
        self.session_expiry = session_expiry


class LoginService:
    """High-level interface to implement a web login service (WLS).

    This class provides a convenient interface for implementing a WLS with any
    authentication backend.  It is intended to be instantiated with a single
    private key, which is used to sign the responses it generates.

    Mechanisms deemed useful for WLS implementation are provided:
      - storing the list of supported authentication methods, and checking
        whether the WLS and a WAA's request have an method in common
      - checking whether the protocol version specified in the WAA request is
        supported by `ucam_wls`

    These mechanisms can optionally be turned off.

    Attributes:
        key (ucam_wls.signing.Key): a private key to be used to sign responses
        auth_methods (list): a list of supported authentication methods
    """
    def __init__(self, key, auth_methods):
        if not isinstance(key, Key):
            raise TypeError("key must be a ucam_wls.signing.Key instance")
        self.key = key
        self.auth_methods = auth_methods

    def have_mutual_auth_type(self, request):
        if request.aauth and any(request.aauth):
            return set(request.aauth) & set(self.auth_methods) != set()
        else:
            return True

    def _pre_response(self, request, skip_handling_check, check_auth_types=True):
        if not skip_handling_check:
            if not request.data_valid:
                raise InvalidAuthRequest
            if check_auth_types and not self.have_mutual_auth_type(request):
                raise NoMutualAuthType(
                    "WLS supports %s; WAA wants one of %s" % (
                        self.auth_methods, request.aauth
                    )
                )
            if not request.version_supported:
                raise ProtocolVersionUnsupported(request.ver)

    def _finish_response(self, response, sign=True, force_signature=False):
        if sign or response.requires_signature:
            if not response.is_signed or force_signature:
                self.key.sign(response)
        return response

    def authenticate_active(self, request, principal, auth, life=None,
                            sign=True, skip_handling_check=False, *args, **kwargs):
        """Generate a WLS 'success' response based on interaction with the user

        This function creates a WLS response specifying that the principal was
        authenticated based on 'fresh' interaction with the user (e.g. input of
        a username and password).

        Args:
            request (AuthRequest): the original WAA request
            principal (AuthPrincipal): the principal authenticated by the WLS
            auth (str): the authentication method used by the principal.
            life (int): if specified, the validity (in seconds) of the
                        principal's session with the WLS.
            sign (bool): whether to sign the response or not.  Recommended to
                leave this at the default value of `True` (see warning below).

            *args: passed to `AuthResponse.respond_to_request`
            **kwargs: passed to `AuthResponse.respond_to_request`

        Returns:
            An `AuthResponse` instance matching the given arguments.

        Warning:
            Responses indicating successful authentication *MUST* be signed by
            the WLS.  It is recommended that you leave `sign` set to `True`, or
            make sure to sign the response manually afterwards.
        """
        self._pre_response(request, skip_handling_check)

        if request.iact == False:
            raise ValueError("WAA demanded passive authentication (iact == 'no')")

        if life is None and principal.session_expiry is not None:
            life = int((principal.session_expiry - datetime.datetime.utcnow()).total_seconds())

        response = AuthResponse.respond_to_request(
            request=request, code=status.SUCCESS, principal=principal.userid,
            auth=auth, ptags=principal.ptags, life=life, *args, **kwargs
        )
        return self._finish_response(response=response, sign=sign)

    def authenticate_passive(self, request, principal, sso=[], sign=True,
                             skip_handling_check=False, *args, **kwargs):
        """Generate a WLS 'success' response based on a pre-existing identity

        This function creates a WLS response specifying that the principal was
        authenticated based on previous successful authentication (e.g. an
        existing WLS session cookie).

        Args:
            request (AuthRequest): the original WAA request
            principal (AuthPrincipal): the principal authenticated by the WLS
            sso (list): a list of strings indicating the authentication methods
                previously used for authentication by the principal.  If an
                empty list is passed, `principal.auth_methods` will be used.
            sign (bool): whether to sign the response or not.  Recommended to
                leave this at the default value of `True` (see warning below).

            *args: passed to `AuthResponse.respond_to_request`
            **kwargs: passed to `AuthResponse.respond_to_request`

        Returns:
            An `AuthResponse` instance matching the given arguments.

        Warning:
            Responses indicating successful authentication *MUST* be signed by
            the WLS.  It is recommended that you leave `sign` set to `True`, or
            make sure to sign the response manually afterwards.
        """
        self._pre_response(request, skip_handling_check)

        if request.iact == True:
            raise ValueError("WAA demanded active authentication (iact == 'yes')")

        if len(sso) == 0:
            sso = principal.auth_methods

        if len(sso) == 0:
            raise ValueError("no authentication methods specified for `sso`")

        if principal.session_expiry is not None:
            life = int((principal.session_expiry - datetime.datetime.utcnow()).total_seconds())
        else:
            life = None

        response = AuthResponse.respond_to_request(
            request=request, code=status.SUCCESS, principal=principal.userid,
            sso=sso, ptags=principal.ptags, life=life, *args, **kwargs
        )
        return self._finish_response(response=response, sign=sign)

    def generate_failure(self, code, request, msg='', sign=True,
                         skip_handling_check=False, *args, **kwargs):
        """Generate a response indicating failure.

        This is to be used in all cases where the outcome of user interaction
        is not success.  This function will refuse to handle a request where
        the 'fail' parameter is 'yes' (in which case the WLS must not redirect
        back to the WAA).

        Args:
            code (int): the response status code.  Values specified in the
                protocol are available as constants under `ucam_wls.status`.
            request (AuthRequest): the original WAA request
            msg (str): an optional message that could be shown to the end user
                by the WAA
            sign (bool): whether to sign the response or not.

            *args: passed to `AuthResponse.respond_to_request`
            **kwargs: passed to `AuthResponse.respond_to_request`

        Returns:
            An `AuthResponse` instance matching the given arguments.

        Note:
            Signatures on WLS responses indicating a non-success can optionally
            be signed.   In the interests of security, the default in this
            function is to go ahead and sign anyway, but this can be turned off
            if really desired.
        """
        self._pre_response(request, skip_handling_check, check_auth_types=False)

        if request.fail:
            raise ValueError("WAA specified that WLS must not redirect "
                             "back to it on failure")

        if code == status.SUCCESS:
            raise ValueError("Failure responses must not have success status")

        response = AuthResponse.respond_to_request(
            request=request, code=code, *args, **kwargs
        )
        return self._finish_response(response=response, sign=sign)
