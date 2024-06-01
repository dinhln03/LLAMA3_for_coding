"""Tcp client for synchronous uhd message tcp port"""

import threading
import Queue
import time
import socket
import struct
import numpy as np

class _TcpSyncClient(threading.Thread):
    """Thead for message polling"""
    queue = Queue.Queue()
    q_quit = Queue.Queue()

    ip_address = None
    port = None

    def __init__(self, ip_address, port, packet_size, packet_type):
        super(_TcpSyncClient, self).__init__()
        self.ip_address = ip_address
        self.port = port
        self.packet_size = packet_size
        self.packet_type = packet_type

    def __exit__(self):
        self.stop()

    def run(self):
        """connect and poll messages to queue"""

        #Establish connection
        sock = None
        print("Connecting to synchronous uhd message tcp port " + str(self.port))
        while self.q_quit.empty():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.ip_address, self.port))
                break
            except socket.error:
                print("connecting to synchronous uhd message tcp port " + str(self.port))
                #traceback.print_exc()
                sock.close()
                time.sleep(0.5)
        print("Connected to synchronous uhd message tcp port " + str(self.port))

        #Read messages
        sock.settimeout(None)
        s = ""
        while self.q_quit.empty():
            try:

                #concatenate to one package
                while self.q_quit.empty():
                    s += sock.recv(self.packet_size)
                    if (len(s)) >= self.packet_size:
                        break
                res_tuple = struct.unpack( self.packet_type, s[:self.packet_size])
                s = s[self.packet_size:]
                self.queue.put(res_tuple)
            except socket.timeout:
                self.stop()
                traceback.print_exc()
                pass

        sock.close()

    def stop(self):
        """stop thread"""
        print("stop tcp_sync uhd message tcp thread")
        self.q_quit.put("end")


class UhdSyncMsg(object):
    """Creates a thread to connect to the synchronous uhd messages tcp port"""

    def __init__(self, ip_address = "127.0.0.1", port = 47009, packet_size = 3, packet_type = "fff"):
        self.tcpa = _TcpSyncClient(ip_address, port, packet_size, packet_type)
        self.tcpa.start()

    def __exit__(self):
        self.tcpa.stop()

    def stop(self):
        """stop tcp thread"""
        self.tcpa.stop()

    def get_msgs(self, num):
        """get received messages as string of integer"""
        out = []
        while len(out) < num:
            out.append(self.tcpa.queue.get())
        return out

    def get_msgs_fft(self, num):
        """
        get received messages as string of integer
        apply fftshift to message
        """
        out = []
        while len(out) < num:
            out.append(self.tcpa.queue.get())
        return [np.fft.fftshift(np.array(o)) for o in out]

    def get_res(self):
        """get received messages as string of integer"""
        out = []
        while not self.tcpa.queue.empty():
            out.append(self.tcpa.queue.get())
        return out

    def has_msg(self):
        """Checks if one or more messages were received and empties the message queue"""
        return self.get_res() != ""
