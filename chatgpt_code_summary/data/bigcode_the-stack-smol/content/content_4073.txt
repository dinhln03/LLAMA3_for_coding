import logging

from openstack_internal.nova.hypervisor_details import OSHypervisor
from topology.link import Link
from topology.node import Node

LOG = logging.getLogger(__name__)


class Server(Node):

    def __init__(self, int_id: int, hypervisor: OSHypervisor):
        super().__init__(int_id=int_id, _id=hypervisor.get_id(), name=hypervisor.get_name(), is_switch=False)
        print(f"Server Name: {self.name}")
        self.cpu = hypervisor.get_available_vcpus()
        self.hdd = hypervisor.get_available_disk_gb()
        self.ram = hypervisor.get_available_ram_mb()
        self.in_links: Link or None = None
        self.out_links: Link or None = None

    def add_in_link(self, link: Link):
        self.in_links = link

    def add_out_link(self, link: Link):
        self.out_links = link

