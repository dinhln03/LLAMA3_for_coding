"""
Module Reader Writer
This module provide the ReaderWriter class as a concrete implemenation of the AbstractReaderWriter. It handles
the implementation details of interfacing with the hardware.
"""

from controlpyweb.abstract_reader_writer import AbstractReaderWriter
import requests
import json
from typing import Union, Optional, List
import time
import threading


from controlpyweb.errors import ControlPyWebAddressNotFoundError, WebIOConnectionError

lock = threading.Lock()

class ReaderWriter(AbstractReaderWriter):

    def __init__(self, url: str, demand_address_exists: bool = True, timeout: float = 10.0,
                 keep_alive: bool = True, **kwargs):
        """
        :param url: The address of the IO Base module from/to which IO is written
        """
        url = 'http://{}'.format(url) if 'http' not in url else url
        url = '{}/customState.json'.format(url)
        self._url = url    # type: str
        self._io = dict()
        self._previous_read_io = dict()
        self._changes = dict()
        self._first_read = False
        self._last_hardware_read_time = None            # type: time.time
        self._req = requests if not keep_alive else requests.Session()
        self.update_reads_on_write = bool(kwargs.get('update_reads_on_write', False))
        self.demand_address_exists = demand_address_exists
        self.timeout = timeout

    @property
    def last_hardware_read_time(self):
        return self._last_hardware_read_time

    def _check_for_address(self, addr: str):
        if not self.demand_address_exists:
            return
        if not self._first_read:
            return
        if self._io is None:
            return
        if addr not in self._io:
            raise ControlPyWebAddressNotFoundError(addr)

    def _get(self, timeout: float = None) -> dict:
        """ Does an http get and returns the results as key/value pairs"""
        timeout = self.timeout if timeout is None else timeout
        self._first_read = True
        r = self._req.get(self._url, timeout=timeout)
        r = None if r is None else r.json()
        return r

    @staticmethod
    def _value_to_str(value):
        if isinstance(value, bool):
            value = '1' if value else '0'
        return str(value)

    @property
    def changes(self):
        """Returns a dictionary of all changes made since the last read or write"""
        return self._changes

    def dumps(self, changes_only: bool = False):
        """Returns the current IO key/values as json string"""
        with lock:
            if changes_only:
                if len(self._changes) == 0:
                    return ''
                return json.dumps(self._changes)
            return json.dumps(self._io)

    def flush_changes(self):
        """ Erases the collection of changes stored in memory"""
        with lock:
            self._changes = dict()

    def loads(self, json_str: str):
        """Replaces the current IO key/values with that from the json string"""
        with lock:
            self._first_read = True
            self._io = json.loads(json_str)

    def read(self, addr: str) -> Optional[Union[bool, int, float, str]]:
        """
        Returns the value of a single IO from the memory store
        """
        with lock:
            if not self._first_read:
                return None
            self._check_for_address(addr)
            val = self._io.get(addr)
            return val

    def read_immediate(self, addr: str, timeout: float = None) -> object:
        """
        Makes a hardware call to the base module to retrieve the value of the IO. This is inefficient and should
        be used sparingly.
        """
        try:
            self._check_for_address(addr)
            timeout = self.timeout if timeout is None else timeout
            vals = self._get(timeout=timeout)
            if vals is None:
                return None
            return vals.get(addr)
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout) as ex:
            raise WebIOConnectionError(ex)

    def to_hardware(self, timeout: float = None):
        """ Same as send_changes_to_hardware"""
        return self.send_changes_to_hardware(timeout)

    def send_changes_to_hardware(self, timeout: float = None):
        """ Takes the collection of changes made using the write command and
        sends them all to the hardware collectively. """
        try:
            with lock:
                if self._changes is None or len(self._changes) == 0:
                    return
                timeout = self.timeout if timeout is None else timeout
                self._req.get(self._url, params=self._changes, timeout=timeout)
            self.flush_changes()
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout) as ex:
            raise WebIOConnectionError(ex)

    def from_hardware(self, timeout: float = None):
        """ Same as update_from_hardware"""
        self.update_from_hardware(timeout)

    def update_from_hardware(self, timeout: float = None):
        """Makes a hardware call to the base module to retrieve the value of all IOs, storing their
        results in memory."""
        try:
            timeout = self.timeout if timeout is None else timeout
            with lock:
                vals = self._get(timeout)
                self._last_hardware_read_time = time.time()
                if vals is not None:
                    self._io = vals
            self.flush_changes()
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout) as ex:
            raise WebIOConnectionError(ex)

    def write(self, addr: str, value: object) -> None:
        """
        Stores the write value in memory to be written as part of a group write when changes are sent to
        hardware."""
        with lock:
            to_str = self._value_to_str(value)
            if self.update_reads_on_write:
                self._io[addr] = value
            self._changes[addr] = to_str

    def write_immediate(self, addr: Union[str, List[str]],
                        value: Union[object, List[object]], timeout: float = None):
        """
        Instead of waiting for a group write, writes the given value immediately. Note, this is not very efficient
        and should be used sparingly. """

        if isinstance(addr, list):
            if isinstance(value, list):
                items = {addr: self._value_to_str(val) for addr, val in zip(addr, value)}
            else:
                value = self._value_to_str(value)
                items = {addr: value for addr in addr}
        else:
            items = {addr: self._value_to_str(value)}

        try:
            timeout = self.timeout if timeout is None else timeout
            with lock:
                self._req.get(self._url, params=items, timeout=timeout)
                for addr, value in items.items():
                    self._io[addr] = value
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout) as ex:
            raise WebIOConnectionError(ex)


