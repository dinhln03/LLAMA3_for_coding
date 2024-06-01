#!/usr/bin/env python
import contextlib as __stickytape_contextlib

@__stickytape_contextlib.contextmanager
def __stickytape_temporary_dir():
    import tempfile
    import shutil
    dir_path = tempfile.mkdtemp()
    try:
        yield dir_path
    finally:
        shutil.rmtree(dir_path)

with __stickytape_temporary_dir() as __stickytape_working_dir:
    def __stickytape_write_module(path, contents):
        import os, os.path

        def make_package(path):
            parts = path.split("/")
            partial_path = __stickytape_working_dir
            for part in parts:
                partial_path = os.path.join(partial_path, part)
                if not os.path.exists(partial_path):
                    os.mkdir(partial_path)
                    with open(os.path.join(partial_path, "__init__.py"), "wb") as f:
                        f.write(b"\n")

        make_package(os.path.dirname(path))

        full_path = os.path.join(__stickytape_working_dir, path)
        with open(full_path, "wb") as module_file:
            module_file.write(contents)

    import sys as __stickytape_sys
    __stickytape_sys.path.insert(0, __stickytape_working_dir)

    __stickytape_write_module('dispatcher.py', b'# Copyright 2021 Gr\xc3\xa9goire Payen de La Garanderie. Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE file.\n\nimport select\nimport socket\nfrom typing import Any, Dict, Union, TextIO, TYPE_CHECKING, Optional, List\n\n\nif TYPE_CHECKING:\n    from processor import Processor\n    from pydev_server_monitor import PydevServerMonitor\n\n\nclass Dispatcher:\n    """\n    The dispatcher class implements the main loop of the program,\n    waiting for new I/O inputs (either from socket or pipe),\n    then calling the relevant processor to handle the input.\n\n    It also regularly calls monitors which are used to perform health checks\n    on Pydev debug servers. If auto_stop is enabled, the loop exits when the last\n    monitor terminates (i.e. no Pydev debug servers are running).\n    """\n    def __init__(self, auto_stop: bool):\n        self._port_to_processors: "Dict[Any, Processor]" = {}\n        self._socket_to_processors: Dict[Union[socket.socket, TextIO], Processor] = {}\n        self._server_monitors: Dict[Any, PydevServerMonitor] = {}\n        self._auto_stop = auto_stop\n\n    def add_processor(self, processor: "Processor"):\n        self._port_to_processors[processor.key] = processor\n        self._socket_to_processors[processor.socket] = processor\n\n    def remove_processor(self, processor: "Processor"):\n        try:\n            del self._port_to_processors[processor.key]\n            del self._socket_to_processors[processor.socket]\n        except KeyError:\n            pass\n        processor.close()\n\n    def add_server_monitor(self, monitor: "PydevServerMonitor"):\n        self._server_monitors[monitor.key] = monitor\n\n    def remove_server_monitor(self, monitor: "PydevServerMonitor"):\n        try:\n            del self._server_monitors[monitor.key]\n        except KeyError:\n            pass\n\n    def find_processor(self, key: Any) -> "Optional[Processor]":\n        return self._port_to_processors.get(key, None)\n\n    def get_all_processors(self) -> "List[Processor]":\n        return list(self._port_to_processors.values())\n\n    def dispatch_loop(self):\n        while True:\n            inputs = list(self._socket_to_processors.keys())\n        \n            inputs_ready, _, _ = select.select(inputs, [], [], 1)\n\n            for input_socket in inputs_ready:\n                processor = self._socket_to_processors[input_socket]\n                processor.on_input_ready()\n\n            for monitor in list(self._server_monitors.values()):\n                monitor.monitor()\n\n            if self._auto_stop and len(self._server_monitors) == 0:\n                return\n  \n')
    __stickytape_write_module('processor.py', b'# Copyright 2021 Gr\xc3\xa9goire Payen de La Garanderie. Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE file.\n\nimport abc\nimport socket\nfrom typing import Any, Union, TextIO\n\n\nclass Processor(abc.ABC):\n    @property\n    @abc.abstractmethod\n    def key(self) -> Any: raise NotImplementedError\n\n    @property\n    @abc.abstractmethod\n    def socket(self) -> Union[socket.socket, TextIO]: raise NotImplementedError\n\n    @abc.abstractmethod\n    def on_input_ready(self) -> None: raise NotImplementedError\n\n    @abc.abstractmethod\n    def close(self) -> None: raise NotImplementedError\n')
    __stickytape_write_module('pydev_server_monitor.py', b'# Copyright 2021 Gr\xc3\xa9goire Payen de La Garanderie. Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE file.\n\nimport logging\nimport socket\nfrom typing import Any\n\nfrom dispatcher import Dispatcher\nfrom pipe_client_server import PipeClientServer\n\nlogger = logging.getLogger("pydev_server_monitor")\n\n\nclass PydevServerMonitor:\n    """\n    Monitor a local Pydev debug server.\n\n    When initialised, this class sends a message to the remote to create a corresponding listening server.\n    When the Pydev server stops, this class detects that the server is no longer running\n    and also close the remote server.\n    """\n    def __init__(self, dispatcher: Dispatcher, local_port: str):\n        logger.debug(f"start monitoring the port {local_port}")\n        self._dispatcher = dispatcher\n        self._local_port = local_port\n        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n        #self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n\n        self._is_terminated = False\n\n        if self.is_socket_alive():\n            server = self._dispatcher.find_processor(None)\n            assert isinstance(server, PipeClientServer)\n\n            logger.debug(f"ask remote to start new server on port {local_port}")\n            server.write(local_port, "", "start_server\\n")\n        else:\n            logger.debug(f"server is not running")\n            self._is_terminated = True\n\n    @property\n    def key(self) -> Any:\n        return self._local_port\n    \n    def is_socket_alive(self) -> bool:\n        if self._is_terminated:\n            return False\n\n        try:\n            self._socket.bind((\'\', int(self._local_port)))\n        except Exception:\n            return True\n\n        try:\n            self._socket.shutdown(2)\n        except:\n            pass\n\n        return False\n\n    def monitor(self):\n        if not self.is_socket_alive() and not self._is_terminated:\n            server = self._dispatcher.find_processor(None)\n            assert isinstance(server, PipeClientServer)\n\n            logger.debug(f"ask remote to stop server on port {self._local_port}")\n            server.write(self._local_port, "", "stop_server\\n")\n            self._dispatcher.remove_server_monitor(self)\n            self._is_terminated = True\n')
    __stickytape_write_module('pipe_client_server.py', b'# Copyright 2021 Gr\xc3\xa9goire Payen de La Garanderie. Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE file.\n\nimport fcntl\nimport logging\nimport os\nimport io\nfrom typing import Any, BinaryIO\n\nfrom dispatcher import Dispatcher\nfrom processor import Processor\n\nlogger = logging.getLogger("pipe_client_server")\n\n\nclass PipeClientServer(Processor):\n    """\n    This class handles the communication between the local and remote hosts using a pipe.\n    """\n    def __init__(self, dispatcher: Dispatcher, stdin: BinaryIO, stdout: BinaryIO):\n        logger.debug("create new pipe client/server")\n        self._dispatcher = dispatcher\n        self._read_buffer = ""\n        self._stdin = stdin\n        self._stdout = stdout\n        orig_fl = fcntl.fcntl(self._stdin, fcntl.F_GETFL)\n        fcntl.fcntl(self._stdin, fcntl.F_SETFL, orig_fl | os.O_NONBLOCK)\n\n    @property\n    def key(self) -> Any:\n        return None\n\n    @property\n    def socket(self) -> BinaryIO:\n        return self._stdin\n\n    def on_input_ready(self):\n        data = self._stdin.read(1024)\n        if len(data) == 0:\n            logger.debug("the end of the pipe has been closed. Exiting.")\n            import sys\n            sys.exit(0)\n\n        self._read_buffer += (data if isinstance(data, str) else data.decode())\n\n        while self._read_buffer.find("\\n") != -1:\n            command, read_buffer = self._read_buffer.split("\\n", 1)\n            self._read_buffer = read_buffer\n\n            args = command.split("\\t", 2)\n\n            local_port = args[0]\n            remote_port = args[1]\n            command = args[2]\n\n            if command == "start_client":\n                self.start_client(local_port, remote_port)\n            elif command == "stop_client":\n                self.close_client(local_port, remote_port)\n            elif command == "start_server":\n                self.start_server(local_port)\n            elif command == "stop_server":\n                self.stop_server(local_port)\n            else:\n                self.dispatch_command_to_client(local_port, remote_port, command+"\\n")\n\n    def write(self, local_port: str, remote_port: str, command: str):\n        data = local_port+"\\t"+remote_port+"\\t"+command\n        if isinstance(self._stdout, (io.BufferedIOBase, io.RawIOBase)):\n            data = data.encode()\n        self._stdout.write(data)\n        self._stdout.flush()\n\n    def start_server(self, local_port: str):\n        logger.debug(f"start the server on {local_port}")\n        from pydev_server import PydevServer\n        server = PydevServer(self._dispatcher, local_port)\n        self._dispatcher.add_processor(server)\n\n    def stop_server(self, local_port: str):\n        logger.debug(f"stop the server on {local_port}")\n        server = self._dispatcher.find_processor(local_port)\n        self._dispatcher.remove_processor(server)\n\n    def start_client(self, local_port: str, remote_port: str):\n        from pydev_client import PydevClient\n        logger.debug(f"create new client (local: {local_port}, remote: {remote_port}")\n        client = PydevClient(self._dispatcher, local_port, remote_port)\n        self._dispatcher.add_processor(client)\n\n    def dispatch_command_to_client(self, local_port: str, remote_port: str, command: str):\n        key = (local_port, remote_port)\n        client = self._dispatcher.find_processor(key)\n        client.write(command)\n\n    def close_client(self, local_port: str, remote_port: str):\n        logger.debug(f"close the client (local: {local_port}, remote: {remote_port})")\n        key = (local_port, remote_port)\n\n        client = self._dispatcher.find_processor(key)\n\n        if client is not None:\n            self._dispatcher.remove_processor(client)\n\n    def close(self) -> None:\n        pass\n')
    __stickytape_write_module('pydev_server.py', b'# Copyright 2021 Gr\xc3\xa9goire Payen de La Garanderie. Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE file.\n\nimport logging\nimport socket\nfrom typing import Any\n\nfrom dispatcher import Dispatcher\nfrom processor import Processor\n\nlogger = logging.getLogger("pydev_server")\n\n\nclass PydevServer(Processor):\n    """\n    Listen on the remote pod for new debugger connection and create a new client for each connection.\n    """\n    def __init__(self, dispatcher: Dispatcher, local_port: str):\n        logger.debug(f"start new server on port {local_port}")\n        self._dispatcher = dispatcher\n        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n        self._socket.bind((\'\', int(local_port)))\n        self._socket.listen(100)\n        self._socket.setblocking(False)\n        self._local_port = str(local_port)\n\n    @property\n    def key(self) -> Any:\n        return self._local_port\n\n    @property\n    def socket(self) -> socket.socket:\n        return self._socket\n    \n    def on_input_ready(self):\n        client_socket, address = self._socket.accept()\n        remote_port = address[1]\n\n        from pydev_client import PydevClient\n        from pipe_client_server import PipeClientServer\n\n        self._dispatcher.add_processor(\n                PydevClient(self._dispatcher, self._local_port, str(remote_port), client_socket))\n        \n        server = self._dispatcher.find_processor(None)\n        assert isinstance(server, PipeClientServer)\n\n        server.write(self._local_port, str(remote_port), "start_client\\n")\n\n    def close(self):\n        self._socket.close()\n')
    __stickytape_write_module('pydev_client.py', b'# Copyright 2021 Gr\xc3\xa9goire Payen de La Garanderie. Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE file.\n\nimport logging\nimport socket\nfrom typing import Any\n\nfrom dispatcher import Dispatcher\nfrom processor import Processor\nfrom pipe_client_server import PipeClientServer\n\nlogger = logging.getLogger("pydev_client")\n\n\nclass PydevClient(Processor):\n    """\n    Client which reads Pydev commands (either on the local or remote) and send them through the pipe\n    to the other end.\n\n    The client also detects when a Pydev debug server starts a new server.\n    When this happens, a monitor is created to handle this new server.\n    (this is part of the support for multiproc in PyCharm)\n    """\n    def __init__(self, dispatcher: Dispatcher, local_port: str, remote_port: str, client_socket=None):\n        logger.debug(f"start new client (local: {local_port}, remote: {remote_port})")\n        self._read_buffer = ""\n        self._dispatcher = dispatcher\n\n        if client_socket is None:\n            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n            self._socket.connect(("127.0.0.1", int(local_port)))\n        else:\n            self._socket = client_socket\n\n        self._socket.setblocking(False)\n        self._local_port = local_port\n        self._remote_port = remote_port\n\n    @property\n    def key(self) -> Any:\n        return self._local_port, self._remote_port\n\n    @property\n    def socket(self) -> socket.socket:\n        return self._socket\n\n    def write(self, data: str):\n        logger.debug("write: "+data)\n        self._socket.sendall(data.encode())\n\n    def on_input_ready(self):\n        server = self._dispatcher.find_processor(None)\n        assert isinstance(server, PipeClientServer)\n\n        recv_data = self._socket.recv(1024).decode()\n        if len(recv_data) == 0:\n            # The socket has been closed\n            logger.debug(f"stop this client, and ask remote to stop (local: {self._local_port}, "\n                         f"remote: {self._remote_port})")\n            server.write(self._local_port, self._remote_port, "stop_client\\n")\n            self._dispatcher.remove_processor(self)\n\n        self._read_buffer += recv_data\n\n        while self._read_buffer.find("\\n") != -1:\n            command, read_buffer = self._read_buffer.split("\\n", 1)\n            self._read_buffer = read_buffer\n\n            # Detect when PyCharm tries to start a new server\n            args = command.split("\\t", 2)\n            if len(args) == 3 and args[0] == "99" and args[1] == "-1":\n                new_local_port = args[2]\n                logger.debug(f"start monitoring for {new_local_port} (local: {self._local_port}, "\n                             f"remote: {self._remote_port})")\n                from pydev_server_monitor import PydevServerMonitor\n                self._dispatcher.add_server_monitor(PydevServerMonitor(self._dispatcher, new_local_port))\n            \n            logger.debug("read : "+command)\n            server.write(self._local_port, self._remote_port, command+"\\n")\n\n    def close(self):\n        self._socket.close()\n')
    # Copyright 2021 Grégoire Payen de La Garanderie. Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE file.
    
    from dispatcher import Dispatcher
    from pipe_client_server import PipeClientServer
    from pydev_server_monitor import PydevServerMonitor
    import sys
    import subprocess
    import os
    
    
    import logging
    
    is_local = len(sys.argv) > 1
    
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    
    format_header = "local" if is_local else "remote"
    formatter = logging.Formatter('%(asctime)s - '+format_header+' %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    
    if is_local:
        #Local connection worker.
        #
        #Start the child connection (the remote), establish the pipe between the parent and child process,
        #then add a monitor for the local Pydev server.
        local_port = sys.argv[1]
        worker_command = sys.argv[2:]
    
        child = subprocess.Popen(worker_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    
        dispatcher = Dispatcher(auto_stop=True)
        dispatcher.add_processor(PipeClientServer(dispatcher, child.stdout, child.stdin))
    
        server_monitor = PydevServerMonitor(dispatcher, local_port)
        if server_monitor.is_socket_alive():
            dispatcher.add_server_monitor(server_monitor)
    else:
        # Remote connection worker.
        #
        # Establish the pipe between the parent and child process.
        dispatcher = Dispatcher(auto_stop=False)
        dispatcher.add_processor(PipeClientServer(dispatcher, sys.stdin, sys.stdout))
        child = None
    
    # Finally, start the main loop
    dispatcher.dispatch_loop()
    
    if child is not None:
        child.terminate()
        child.wait()
    