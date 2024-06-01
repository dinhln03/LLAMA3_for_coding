from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from .redirect import RedirectHandler
import threading
import ssl

__all__ = ['ThreadedServer', 'SecureServer']


class ThreadedServer(ThreadingMixIn, HTTPServer):
    protocol_version = 'HTTP/1.1'

    def __init__(self,
                 host: str,
                 port: int,
                 RequestHandlerClass: BaseHTTPRequestHandler,
                 bind_and_activate: bool=True):
        self._serve_forever_thread = None  # type: threading.Thread
        super().__init__((host, port), RequestHandlerClass, bind_and_activate)

    def serve_forever(self, poll_interval=0.5):
        self._serve_forever_thread = threading.Thread(
            target=super().serve_forever,
            args=(poll_interval,)
        )
        self._serve_forever_thread.start()


class SecureServer(ThreadedServer):
    def __init__(self,
                 certfile: str,
                 keyfile: str,
                 host: str,
                 port: int,
                 RequestHandlerClass: BaseHTTPRequestHandler,
                 bind_and_activate: bool = True):
        self._certfile = certfile
        self._keyfile = keyfile
        self._redirect = ThreadedServer(host,
                                        80,
                                        RedirectHandler,
                                        bind_and_activate)
        super().__init__(host, port, RequestHandlerClass, bind_and_activate)

    def server_bind(self):
        super().server_bind()
        self._redirect.server_bind()
        self.socket = ssl.wrap_socket(self.socket,
                                      server_side=True,
                                      certfile=self._certfile,
                                      keyfile=self._keyfile,
                                      do_handshake_on_connect=False)

    def get_request(self):
        sock, addr = super().get_request()
        sock.do_handshake()
        return sock, addr

    def serve_forever(self, poll_interval=0.5):
        super().serve_forever(poll_interval)
        self._redirect.serve_forever(poll_interval)

    def shutdown(self):
        super().shutdown()
        self._redirect.shutdown()
