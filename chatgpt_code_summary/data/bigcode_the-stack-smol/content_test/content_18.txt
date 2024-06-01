# -*- coding: utf-8 -*-
from app.libs.utils import data_decode
import socket, socketserver, threading
import traceback

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    ip = ""
    port = 0
    timeOut = 100

    def __init__(self, request, client_address, server):
        from app.service.device import Device
        self.socket = None
        self.addr = None
        self.cloud_id = None
        self.device = Device()
        self.sign = None
        self.device_id = None
        self.timestamp = None
        super().__init__(request, client_address, server)

    def setup(self):
        self.ip = self.client_address[0].strip()
        self.port = self.client_address[1]
        self.request.settimeout(self.timeOut)
        self.addr = self.ip + str(self.port)
        self.socket = self.request
        print(self.ip)

    def handle(self):
        try:
            while True:
                try:
                    # time.sleep(1)
                    data = self.request.recv(1024)
                except socket.timeout:
                    print(self.ip + ":" + str(self.port) + "接收超时")
                    break
                if data:
                    data = data_decode(data)
                    self.device.parse_data(data, self)
                else:
                    break
        except Exception as e:
            with open("err_log.log", "a+") as f:
                f.write(traceback.format_exc()+'\r\r')
            print(self.client_address, "连接断开")
        finally:
            self.request.close()

    def finish(self):
        if self.cloud_id is None:
            print(self.ip + ":" + str(self.port) + "断开连接！")
        else:
            get_instance().remove_client(self.cloud_id)
            print(self.ip + ":" + str(self.port) + self.cloud_id + "断开连接！")


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


class TCPServer:
    instance = None

    @staticmethod
    def get_instance():
        print("start")
        if TCPServer.instance is None:
            TCPServer.instance = TCPServer()
        return TCPServer.instance

    def __init__(self):
        self.clients = {}
        self.server = None
        try:
            self.server = ThreadedTCPServer(("0.0.0.0", 5002), ThreadedTCPRequestHandler)
            server_thread = threading.Thread(target=self.server.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            # server_thread.join()
        except (KeyboardInterrupt, SystemExit, Exception) as e:
            print(e)
            print("end")
            self.server.shutdown()
            self.server.close()

    def add_client(self, cloud, sock):
        self.clients[cloud] = sock
        print("this is clients", self.clients)

    def remove_client(self, cloud):
        if cloud in self.clients:
            print("删除设备" + cloud)
            from app.service.device import Device
            Device.offline_alarm(self.clients[cloud])
            self.clients.pop(cloud)


def get_instance():
    return TCPServer.get_instance()

