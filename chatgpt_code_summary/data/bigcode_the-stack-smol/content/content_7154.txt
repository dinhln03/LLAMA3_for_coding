import collections
import itertools
import re
import yaml

from ansible.errors import AnsibleError
from cumulus_vxconfig.utils.filters import Filters
from cumulus_vxconfig.utils import File, Network, Link, Inventory, Host

filter = Filters()
mf = File().master()
inventory = Inventory()


class CheckVars:
    '''
    Pre-Check of variables defined in master.yml
    '''
    def _yaml_f(self, data, style="", flow=None, start=True):
        return yaml.dump(
            data, default_style=style,
            explicit_start=start,
            default_flow_style=flow
        )

    @property
    def vlans(self):
        master_vlans = mf['vlans']

        vids = []
        for tenant, vlans in master_vlans.items():
            for vlan in vlans:
                vids.append(vlan['id'])

        dup_vids = [vid for vid in set(vids) if vids.count(vid) > 1]
        if len(dup_vids) > 0:
            error = {}
            for tenant, value in master_vlans.items():
                v = [v for v in value for d in dup_vids if v['id'] == d]
                if len(v) > 0:
                    error[tenant] = v
            msg = ("VLANID conflict:\nRefer to the errors "
                   "below and check the 'master.yml' file\n{}")
            raise AnsibleError(
                msg.format(self._yaml_f({'vlans': error}))
            )

        return master_vlans

    @property
    def mlag_bonds(self):
        mlag_bonds = mf['mlag_bonds']

        vids = {}
        for tenant, vlans in self.vlans.items():
            ids = []
            for vlan in vlans:
                ids.append(vlan['id'])
                vids[tenant] = ids

        vids_list = [x for i in vids.values() for x in i]

        def _get_tenant(vid):
            return ''.join([
                k for k, v in self.vlans.items() for i in v if i['id'] == vid
            ])

        for rack, bonds in mlag_bonds.items():

            # Check for bonds member conflict
            mems = filter.uncluster(
                [i for v in bonds for i in v['members'].split(',')]
                )
            for mem in set(mems):
                if mems.count(mem) > 1:
                    self._mlag_bonds_error(
                        rack, mem, 'bond member conflict: ' + mem
                    )

            # Check for bonds name conflict
            names = [v['name'] for v in bonds]
            for name in set(names):
                if names.count(name) > 1:
                    self._mlag_bonds_error(
                        rack, name, 'bond name conflict: ' + name
                    )

            for bond in bonds:
                set_items = set([])
                bond_vids = filter.uncluster(bond['vids'])

                for bond_vid in bond_vids:
                    # Check if assign vids exist in tenant vlans

                    if bond_vid not in vids_list:
                        self._mlag_bonds_error(
                            rack, bond['vids'], 'VLANID not found: ' + bond_vid
                        )

                if len(bond_vids) > 1:
                    for bond_vid in bond_vids:
                        set_items.add((_get_tenant(bond_vid), bond['vids']))

                if len(set_items) > 1:
                    title = ("bond assigned with a VLANID which "
                             "belongs to multiple tenant: ")
                    for item in set_items:
                        self._mlag_bonds_error(rack, item[1], title)

        return mlag_bonds

    def _mlag_bonds_error(self, rack, item, title):
        bonds = []
        for bond in mf['mlag_bonds'][rack]:
            if item in filter.uncluster(bond['members']):
                bonds.append(bond)
            elif bond['name'] == item:
                bonds.append(bond)
            elif bond['vids'] == item:
                bonds.append(bond)

        if len(bonds) > 0:
            msg = ("{}\nRefer to the errors below and "
                   "check the 'master.yml' file.\n{}")

            raise AnsibleError(msg.format(
                title, filter.yaml_format({'mlag_bonds': {rack: bonds}})
                )
            )

    @property
    def mlag_peerlink_interfaces(self):
        mlag_peerlink_interfaces = mf['mlag_peerlink_interfaces']
        ifaces = filter.uncluster(mlag_peerlink_interfaces)

        dup_ifaces = [i for i in set(ifaces) if ifaces.count(i) > 1]
        if len(dup_ifaces) > 0:
            msg = ("interfaces conflict:\nRefer to the errors below and "
                   "check the 'master.yml' file.\n{}")
            raise AnsibleError(
                msg.format(self._yaml_f({
                    'mlag_peerlink_interfaces': mlag_peerlink_interfaces
                }, flow=False)))

        return ','.join(ifaces)

    @property
    def base_networks(self):
        base_networks = mf['base_networks']

        def networks():
            networks = collections.defaultdict(list)
            for k, v in base_networks.items():
                if isinstance(v, dict):
                    for _k, _v in v.items():
                        net = Network(_v)
                        networks[net.id].append((k, _k, net))
                else:
                    net = Network(v)
                    networks[net.id].append((k, net))

            _networks = []
            for k, v in networks.items():
                _networks.extend(list(itertools.combinations(v, 2)))

            return _networks

        def overlaps():
            for items in networks():
                nets = items[0][-1], items[1][-1]
                net_a, net_b = sorted(nets)
                if net_a.overlaps(net_b):
                    return items

        if overlaps() is not None:
            error = collections.defaultdict(dict)
            for item in overlaps():
                if len(item) > 2:
                    error[item[0]][item[1]] = str(item[-1])
                else:
                    error[item[0]] = str(item[-1])
            msg = ("networks conflict:\nRefer to the errors below and "
                   "check the 'master.yml' file.\n{}")
            raise AnsibleError(
                msg.format(self._yaml_f(
                    {'base_networks': dict(error)}, flow=False))
            )

        return base_networks

    def vlans_network(self, tenant, vlan, vlans_network=None):
        vnp = Network(vlan['network_prefix'])

        # Check vlan subnet against base_networks
        for var, net in self.base_networks.items():
            if var != 'vlans':
                if var == 'loopbacks':
                    for group, network in net.items():
                        _net = Network(network)
                        if (vnp.overlaps(_net)
                                or _net.overlaps(vnp)):
                            msg = ("networks conflict:\nRefer to the errors "
                                   "below and check the 'master.yml' file."
                                   "\n{}\n{}")
                            raise AnsibleError(msg.format(
                                filter.yaml_format({
                                    'vlans': {tenant: [vlan]}
                                }),
                                filter.yaml_format({
                                    'base_networks': {var: {group: network}}
                                }, start=False)
                            ))
                else:
                    _net = Network(net)
                    if (vnp.overlaps(_net)
                            or _net.overlaps(vnp)):
                        msg = ("networks conflict:\nRefer to the errors below "
                               "and check the 'master.yml' file.\n{}\n{}")
                        raise AnsibleError(msg.format(
                            filter.yaml_format({'vlans': {tenant: [vlan]}}),
                            filter.yaml_format({
                                'base_networks': {var: net}}, start=False)
                        ))

        for k, v in vlans_network.items():
            _vnp = Network(v['network_prefix'])
            if (vnp.overlaps(_vnp)
                    or _vnp.overlaps(vnp)):
                msg = ("networks conflict: {} overlaps with existing network "
                       "{}(VLAN{})\nRefer to the errors below and check the "
                       "'master.yml' file.\n{}")
                raise AnsibleError(msg.format(
                    str(vnp), str(_vnp), v['id'],
                    filter.yaml_format({'vlans': {tenant: [vlan]}})
                ))

    def link_base_network(self, name):
        base_networks = self.base_networks
        if name not in base_networks:
            raise AnsibleError(
                "Please define a base networks for network link '{}' in "
                "'base_networks' variable in 'master.yml'".format(name)
            )

        return base_networks[name]

    @property
    def base_asn(self):
        base_asn = mf['base_asn']
        group_asn = ((k, v) for k, v in base_asn.items())
        for group_asn in itertools.combinations(group_asn, 2):
            g1, g2 = group_asn
            if g1[1] == g2[1]:
                msg = ("duplicate AS: {}\n"
                       "Refer to the errors below and check the "
                       "'master.yml' file.\n{}")
                x = {item[0]: item[1] for item in group_asn}
                raise AnsibleError(
                    msg.format(g1[1], filter.yaml_format({'base_asn': x}))
                )

        for k, _ in base_asn.items():
            if k not in inventory.group_names():
                raise AnsibleError(
                    'Group not found: {}'.format(k)
                )

        return base_asn

    @property
    def interfaces(self):
        interfaces = collections.defaultdict(set)

        # Interfaces in links
        net_links = mf['network_links']
        for k, v in net_links.items():
            links = Link(k, v['links'])
            device_interfaces = links.device_interfaces()
            for dev in device_interfaces:
                re_order = []
                for item in device_interfaces[dev]:
                    port, link, var, name = item
                    re_order.append((port, var, name, link))

                for item in re_order:
                    interfaces[dev].add(tuple(item))

        # Interface in mlag bonds
        mlag_bonds = self.mlag_bonds
        for rack, bonds in mlag_bonds.items():
            items = []
            for idx, bond in enumerate(bonds):
                members = filter.uncluster(bond['members'])
                for member in members:
                    items.append((member, 'mlag_bonds', rack, idx))

            for host in inventory.hosts('leaf'):
                _host = Host(host)
                if rack == _host.rack:
                    for item in items:
                        interfaces[host].add(tuple(item))

        # Interfaces peerlink
        ifaces = filter.uncluster(mf['mlag_peerlink_interfaces'])
        for host in inventory.hosts('leaf'):
            for iface in ifaces:
                item = (
                    iface,
                    'mlag_peerlink_interfaces',
                    mf['mlag_peerlink_interfaces']
                )
                interfaces[host].add(tuple(item))

        hosts = [h for h in interfaces if h in inventory.hosts()]
        for host in hosts:
            x = interfaces[host]
            for k, v in interfaces.items():
                if host in inventory.hosts(k):
                    interfaces[host] = interfaces[k] | x

        for k, v in interfaces.items():
            ports = [item[0] for item in v]
            dup_ports = [item for item in set(ports) if ports.count(item) > 1]
            if len(dup_ports):
                error_items = [i for i in v if i[0] == dup_ports[0]]
                yaml_vars = collections.defaultdict(dict)
                for item in error_items:
                    port, mfvar, name, *r = item
                    if mfvar == 'network_links':
                        yaml_vars[mfvar][name] = {'links': [r[0]]}
                    elif mfvar == 'mlag_bonds':
                        yaml_vars[mfvar][name] = mf[mfvar][name][r[0]]
                    elif mfvar == 'mlag_peerlink_interfaces':
                        yaml_vars[mfvar] = name

                _yaml_vars = {}
                for k, v in yaml_vars.items():
                    if isinstance(v, collections.defaultdict):
                        _yaml_vars.update({k: dict(v)})
                    else:
                        _yaml_vars.update({k: v})

                msg = ("overlapping interface: '{}'\n"
                       "Refer to the errors below and check the "
                       "'master.yml' file.\n{}")

                raise AnsibleError(
                    msg.format(port, filter.yaml_format(_yaml_vars))
                )

    def _mlag_bonds(self, key='name'):
        mlag_bonds = self.mlag_bonds

        x = {}
        for rack, bonds in mlag_bonds.items():
            _groupby = {}
            for k, v in itertools.groupby(bonds, lambda x: x[key]):
                for item in list(v):
                    _groupby[k] = item

            x[rack] = _groupby

        return x

    def _server_bonds(self, key='name'):
        host_ifaces = mf['server_interfaces']
        mlag_bonds = self._mlag_bonds()
        servers = inventory.hosts('server')

        is_rack_exist = True
        is_bond_exist = True

        _server_bonds = {}
        for host, iface in host_ifaces.items():
            if host in servers:
                for idx, bond in enumerate(iface['bonds']):
                    rack = 'rack' + str(bond['rack'])
                    try:
                        mlag_bonds[rack]
                    except KeyError:
                        is_rack_exist = False

                    try:
                        vids = mlag_bonds[rack][bond['name']]['vids']
                    except KeyError:
                        is_bond_exist = False

                    if not is_rack_exist:
                        raise AnsibleError(
                            'rack not found: {} ({}) in {}'.format(
                                rack, list(mlag_bonds.keys()), host)
                            )
                    elif not is_bond_exist:
                        bonds = list(mlag_bonds[rack].keys())
                        raise AnsibleError(
                            'bond not found: {} ({}) in {}'.format(
                                bond['name'], bonds, host)
                            )
                    bond.update({'vids': vids, 'index': idx})

                _groupby = collections.defaultdict(list)
                for k, v in itertools.groupby(iface['bonds'], lambda x: x[key]):
                    list_v = list(v)
                    if key == 'slaves' or key == 'vids':
                        for item in filter.uncluster(k):
                            _groupby[item].extend(list_v)
                    else:
                        _groupby[k].extend(list_v)

                    _server_bonds[host] = _groupby
            else:
                raise AnsibleError(
                    "%s not found in inventory file, "
                    "check the master.yml in server_interfaces" % host
                )

        return _server_bonds

    def server_bonds(self):
        server_bonds = self._server_bonds()
        mlag_bonds = self._mlag_bonds()

        bonds = []

        for host in server_bonds:
            for bond, v in server_bonds[host].items():
                # Check for duplicate bonds per host
                if len(v) > 1:
                    msg = 'multiple bond assignment: {} in {}'
                    raise AnsibleError(msg.format(bond, host))

                for item in v:
                    rack = 'rack' + str(item['rack'])
                    # Check if bond exist in rack per host
                    if item['name'] not in mlag_bonds[rack]:
                        racks = list(mlag_bonds[rack].keys())
                        raise AnsibleError(
                            "bond not found: {} ({}) in {}".format(
                                item['name'], racks, host)
                        )
                    item.update({'host': host})
                bonds.extend(v)

            # Check for duplicate bond slaves per host
            for slave, v in self._server_bonds(key='slaves')[host].items():
                if len(v) > 1:
                    bonds = [i['name'] for i in v]
                    msg = "multiple bond slaves assignment: {} ({}) in {}"
                    raise AnsibleError(msg.format(slave, bonds, host))

            # Check for multiple vlan assignments per host
            for vid, v in self._server_bonds(key='vids')[host].items():
                if len(v) > 1:
                    msg = "multiple bond vlans assignment: {} ({}) in {}"
                    bonds = [i['name'] for i in v]
                    raise AnsibleError(msg.format('vlan' + vid, bonds, host))

        # Check for multiple bond assignment per rack
        for k, v in itertools.groupby(bonds, lambda x: x['rack']):
            list_v = list(v)
            y = collections.defaultdict(list)
            for item in list_v:
                y[item['name']].append(item)

            for bond, _v in y.items():
                if len(_v) > 1:
                    hosts = [i['host'] for i in _v]
                    msg = ('multiple bond assigment '
                           'with the same rack: {} in ({})')
                    raise AnsibleError(msg.format(bond, hosts))

        return server_bonds
