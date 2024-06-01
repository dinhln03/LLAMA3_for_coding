#!/usr/bin/python
# coding: utf-8
"""A simple webserver."""

# python 2.7 compatibility
from __future__ import print_function, unicode_literals

# based on tornado
import tornado.ioloop
import tornado.web
import tornado.websocket

import sys
import json

def make_app():
    """Create and return the main Tornado web application.
    
    It will listen on the port assigned via `app.listen(port)`,
    and will run on Tornado's main ioloop,
    which can be started with `tornado.ioloop.IOLoop.current().start()`.
    """
    return tornado.web.Application([
        (r"/connect", ClientSocket),
        (r"/(.*)", tornado.web.StaticFileHandler, {
            "path": "client",
            "default_filename": "index.html"
        }),
    ], debug=True)

class ClientSocket(tornado.websocket.WebSocketHandler):
    """ClientSocket represents an active websocket connection to a client.
    """
    
    def open(self):
        """Called when a websocket connection is initiated."""
        
        # print some info about the opened connection
        print("WebSocket opened",
              "from user at {}".format(self.request.remote_ip))
    
    def on_message(self, message):
        """Called when a websocket client sends a message."""
        
        # print the message to the console
        print("client sent: {!r}".format(message))
        
        # try to parse the message
        try:
            parsed_message = json.loads(message)
        except ValueError:
            print("Failed to parse message: {!r}".format(message))
            return
        
        # if there's a "message" in the message, echo it
        if "message" in parsed_message:
            response = {
                "client" : str(self.request.remote_ip),
                "message" : parsed_message["message"]
            }
            # respond to the message
            m = json.dumps(response)
            self.write_message(m)
        else:
            print("message unhandled.")
    
    def on_close(self):
        """Called when a client connection is closed for any reason."""
        
        # print some info about the closed connection
        print("WebSocket closed",
              "by user at {}".format(self.request.remote_ip))
        print("close code: {}".format(self.close_code))
        print("close reason: {!r}".format(self.close_reason))

if __name__ == "__main__":
    # print some basic info about the system
    print("Running Tornado Web Server {}".format(tornado.version))
    print("Using Python {}".format(sys.version))
    
    # start the webapp on port 8888
    app = make_app()
    app.listen(8888)
    print("webapp started on port 8888")
    tornado.ioloop.IOLoop.current().start()
