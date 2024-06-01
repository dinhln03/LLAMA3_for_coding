"""
Copyright 2021 K.M Ahnaf Zamil

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

from urllib.parse import parse_qs


import typing

__all__: typing.Final = ["Request", "ImmutableDict", "_Redirect", "Endpoint"]


class _Redirect(object):
    """Just an object for simulating a redirect"""

    def __init__(self, url: str) -> None:
        self.url = url


class ImmutableDict(dict):
    """An immutable dictionary implementation for query arguments and form data"""

    def __setitem__(self, k, v) -> None:
        raise ValueError("ImmutableDict object cannot be modified (immutable)")


class Request(object):
    """An object that contains information related to the HTTP request"""

    def __init__(self, environ):
        self._environ = environ

    @property
    def method(self) -> str:
        """HTTP method used for the request"""
        return self._environ["REQUEST_METHOD"]

    @property
    def endpoint(self) -> str:
        """The route/endpoint used for that specific request"""
        return self._environ["PATH_INFO"]

    @property
    def query_args(self) -> ImmutableDict:
        """Query arguments from the request"""
        args = self._environ["QUERY_STRING"]
        if not args:
            return ImmutableDict({})
        args = args.split("&")
        query_args = {}
        for _arg in args:
            name, value = _arg.split("=")
            query_args[name] = value

        return ImmutableDict(query_args)

    @property
    def form(self) -> typing.Optional[typing.Dict]:
        """Form data sent via HTTP request"""
        data = self._environ.get("wsgi.input")  # Returns io.BytesIO object
        if data:
            form_dict = parse_qs(data.getvalue().decode("utf-8"))

            final_dict = {}

            for k, v in form_dict.items():
                final_dict[k] = v[0]  # Since v is list containing the form data
            return ImmutableDict(final_dict)

    def __str__(self):
        return f'<Request endpoint="{self.endpoint}" method="{self.method}">'


class Endpoint(object):
    def __init__(self, route, func) -> None:
        self.route = route
        self.extension = None
        self._func = func

    def __call__(self, request: Request):
        return self._func(request)
