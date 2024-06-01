#!/usr/bin/env python

# Copyright 2015 Rackspace, Inc
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

from lib import script


class DecomPort(script.TeethScript):
    use_ironic = True
    use_neutron = True

    def __init__(self):
        super(DecomPort, self).__init__(
            'Utility for temporarily putting a node on the decom network.')
        self.add_ironic_node_arguments()
        self.add_argument('command',
                          help='Run command',
                          choices=['add', 'remove'])

    def run(self):

        uuid = self.get_argument('node_uuid')
        node = self.ironic_client.get_node(uuid)

        command = self.get_argument('command')

        if command == 'add':
            self.neutron_client.add_decom_port(node)
        elif command == 'remove':
            self.neutron_client.remove_decom_port(node)


if __name__ == "__main__":
    DecomPort().run()
