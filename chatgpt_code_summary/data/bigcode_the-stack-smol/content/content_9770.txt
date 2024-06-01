import re
import traceback
import subprocess
from serviceDB import ServiceDB


class NeadmServiceWrapper:
    _service_list_cmd = ['/opt/nedge/neadm/neadm', 'service', 'list']

    # _status_cmd = ['/opt/nedge/neadm/fake-neadm-status.sh']
    _service_list_header = re.compile("^.*TYPE.*NAME.*SERVERID.*STATUS.*$")
    # unit_id key well be added during parsing of each line
    _service_list_names = ['type', 'name', 'sid', 'status']

    def __init__(self, db):
        self.exit_code = 0
        self.db = ServiceDB(db)

    def get_exit_code(self):
        return self.exit_code

    def get_raw_output(self, command):
        try:
            output = subprocess.check_output(command, stderr=subprocess.STDOUT)
            self.exit_code = 0
            return output
        except subprocess.CalledProcessError as ex:
            self.exit_code = ex.returncode
            return ex.output
        except Exception as e:
            self.exit_code = 1
            return "Failed to start {0} command.' \
                     Exeption {1}".format(command, e.output)

    def get_all_services(self):
        output = self.get_raw_output(NeadmServiceWrapper._service_list_cmd)
        # print(output)
        result = NeadmServiceList()
        # error exit code
        if self.exit_code:
            result.exit_code = self.exit_code
            result.output = output
            return result

        output_array = output.split('\n')
        for line in output_array:
            # print(line)
            if NeadmServiceWrapper._service_list_header.match(line):
                continue

            params = line.split()
            # print(params)
            # print(len(params))
            if len(params) < 4:
                continue

            service_record = {}

            for name in NeadmServiceWrapper._service_list_names:
                service_record[name] = params[
                    NeadmServiceWrapper._service_list_names.index(name)]

            # check ServiceDB for sid and unit_id already joined
            # add unit_id key
            db_record = self.db.find(sid=service_record['sid'],
                                     service_name=service_record['name'])

            if len(db_record) == 1:
                service_record['unit_id'] = db_record[0]['unit_id']
            else:
                service_record['unit_id'] = ''

            # print(node)
            result.append(service_record)

        # print(status)
        return result

    def exec_cmd(self, cmd_name, cmd):
        try:
            print("\t{0} cmd is {1}".format(cmd_name, ' '.join(cmd)))
            subprocess.check_output(cmd)

        except Exception as ex:
            raise Exception('in {0}\nMessage:{1}\nTrace: {2}'.format(
                self.__class__.__name__, ex.message, traceback.format_exc()))

    # is node included into service nodes list
    def is_node_exist(self, service_name, sid):
        services = self.get_all_services()
        return services.is_already_in_service(service_name, sid)

    # is iscsi service already created
    def is_service_exist(self, service_name):
        services = self.get_all_services()
        return services.is_service_exist(service_name)

    # create new iscsi(cinder) service by name
    def create_iscsi_service(self, service_name):
        cmd = ['/opt/nedge/neadm/neadm', 'service', 'create', 'iscsi',
               service_name]
        if not self.is_service_exist(service_name):
            self.exec_cmd('create_iscsi_service', cmd)
        else:
            print("create_iscsi_service: Service {} already exist!".format(
                service_name))

    # create new swift service by name
    def create_swift_service(self, service_name):
        cmd = ['/opt/nedge/neadm/neadm', 'service', 'create', 'swift',
               service_name]
        if not self.is_service_exist(service_name):
            self.exec_cmd('create_swift_service', cmd)
        else:
            print("create_swift_service: Service {} already exist!".format(
                service_name))

    # remove iscsi service by name
    def delete_service(self, service_name):
        cmd = ['/opt/nedge/neadm/neadm', 'service', 'delete', service_name]
        if self.is_service_exist(service_name):
            self.exec_cmd('delete_service', cmd)
        else:
            print("remove_iscsi_service: {0} service does not exist".format(
                service_name))

    def is_service_enabled(self, service_name):
        services = self.get_all_services()
        return services.is_service_enabled(service_name)

    # serve command, apply swift servie to cluster
    def serve_service(self, service_name, cluster_name):
        cmd = ['/opt/nedge/neadm/neadm', 'service', 'serve', service_name,
               cluster_name]
        if not self.is_service_exist(service_name):
            print("serve_service: Service {} does not exist".format(
                service_name))
            return

        self.exec_cmd('serve_service', cmd)

    # enable service if exist
    def enable_service(self, service_name):
        cmd = ['/opt/nedge/neadm/neadm', 'service', 'enable', service_name]
        if not self.is_service_exist(service_name):
            print("enable_service: Service {} does not exist".format(
                service_name))
            return

        if not self.is_service_enabled(service_name):
            self.exec_cmd('enable_service', cmd)
        else:
            print("enable_service: Service {} already enabled".format(
                service_name))

    def disable_service(self, service_name):
        cmd = ['/opt/nedge/neadm/neadm', 'service', 'disable', service_name]

        if not self.is_service_exist(service_name):
            print("disable_service: Service {} does not exist".format(
                service_name))
            return

        if self.is_service_enabled(service_name):
            self.exec_cmd('disable_service', cmd)
        else:
            print("disable_service: Service {} already disabled".format(
                service_name))

    def add_node_to_service(self, service_name, sid, unit_id):
        cmd = ['/opt/nedge/neadm/neadm', 'service', 'add', service_name, sid]
        if not self.is_node_exist(service_name, sid):
            self.exec_cmd('add_node_to_service', cmd)

            # add node to persistent db
            # self.db.add(sid, unit_id, service_name)
        else:
            print("\tadd_node_to_service:"
                  "Node {0} already exist as service node".format(sid))

        self.db.add(sid, unit_id, service_name)

    def get_service_node_count(self, service_name):
        services = self.get_all_services()
        return len(services.get_service_nodes(service_name))

    def remove_node_by_unit_id(self, unit_id):
        service = self.db.find(unit_id=unit_id)
        if len(service) > 0:
            sid = service[0]['sid']
            service_name = service[0]['service']
            self.remove_node_from_service(service_name, sid, unit_id)
        else:
            print("Can't find service by unit_id:{}".format(unit_id))

    def disable_service_by_unit_id(self, unit_id):
        service = self.db.find(unit_id=unit_id)
        if len(service) > 0:
            service_name = service[0]['service']
            print("service to disable is :{}".format(service_name))
            self.disable_service(service_name)
        else:
            print("Can't find service by unit_id:{}".format(unit_id))

    def remove_node_from_service(self, service_name, sid, unit_id):
        cmd = ['/opt/nedge/neadm/neadm', 'service', 'remove', service_name,
               sid]
        if self.is_node_exist(service_name, sid):
            self.exec_cmd('remove_node_from_service', cmd)

            node_count = self.get_service_node_count(service_name)
            if node_count == 0:
                self.delete_service(service_name)

        else:
            print("\tremove_node_from_service: "
                  "Node {} does not exist to remove".format(sid))

        # remove from persistent db
        self.db.remove(sid, unit_id)

    def print_services(self):
        service_list = self.get_all_services()
        service_list.show()


class NeadmServiceList:
    def __init__(self):
        # service records array
        self.service_records = []
        self.exit_code = 0
        self.output = ""

    def is_correct(self):
        return True if self.exit_code == 0 else False

    def get_all(self):
        return self.service_records

    def get_service_nodes(self, service_name):
        return filter(lambda service: service['name'] == service_name and
                      service['sid'] != '-',
                      self.service_records)

    def get_iscsi_nodes(self):
        return filter(lambda service: service['type'] == 'iscsi' and
                      service['sid'] != '-',
                      self.service_records)

    def get_iscsi_nodes_by_service_name(self, service_name):
        return filter(lambda service: service['type'] == 'iscsi' and
                      service['name'] == service_name and
                      service['sid'] != '-',
                      self.service_records)

    def get_swift_nodes(self):
        return filter(lambda service: service['type'] == 'swift' and
                      service['sid'] != '-',
                      self.service_records)

    def get_swift_nodes_by_service_name(self, service_name):
        return filter(lambda service: service['type'] == 'swift' and
                      service['name'] == service_name and
                      service['sid'] != '-',
                      self.service_records)

    # is node present into whole services list
    def is_already_listed(self, sid):
        return True if filter(lambda service: service['sid'] == sid,
                              self.service_records) else False

    # is node presented in service already
    def is_already_in_service(self, service_name, sid):
        return True if filter(lambda service: service['sid'] == sid and
                              service['name'] == service_name,
                              self.service_records) else False

    def is_service_exist(self, service_name):
        return True if filter(lambda service: service['name'] == service_name,
                              self.service_records) else False

    def is_service_enabled(self, service_name):
        nodes = self.get_service_nodes(service_name)
        print(nodes)
        if len(nodes) > 0:
            if nodes[0]['status'] == 'enabled':
                return True
        return False

    def append(self, service_record):
        self.service_records.append(service_record)

    # def show(self):
    #    print('TYPE\t\tNAME\t\t\tID\t\t\tSTATE\t\t\tUNIT_ID')
    #    for record in self.service_records:
    #        print("{0:<{col0}}{1:<{col1}}{2:<{col2}}"+
    #             "{3:<{col3}}{4:<{col4}}".format(
    #              record['type'],
    #              record['name'],
    #              record['sid'],
    #              record['status'],
    #              record['unit_id'],
    #              col0=8,
    #              col1=20,
    #              col2=36,
    #              col3=12,
    #              col4=16))
    #    print("")
