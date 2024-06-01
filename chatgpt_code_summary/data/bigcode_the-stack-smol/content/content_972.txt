#https://blog.csdn.net/orangefly0214/article/details/81387077
import MultiTemplate
from MultiTemplate import TaskTemplate
# https://blog.csdn.net/u013812710/article/details/72886491
# https://blog.csdn.net/ismr_m/article/details/53100896
#https://blog.csdn.net/bcfdsagbfcisbg/article/details/78134172
import kubernetes
import os
import influxdb
import time

import yaml
def check_path(name):
    train_dir = os.path.join('/tfdata/k8snfs/', name)
    print(train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    return train_dir

def check_ns(name):
    kubernetes.config.load_kube_config()
    v1 = kubernetes.client.CoreV1Api()
    # v1.create_namespace()
    exist_ns = v1.list_namespace()
    exist_ns_name = []
    for i in exist_ns.items:
        exist_ns_name.append(i.metadata.name)
    if name in exist_ns_name:
        return True
    else:
        return False


class SubTask():
    def __init__(self,template_id,ps_replicas,worker_replicas,training_step,batch_size,interval,task_id,rtimes,tag):
        self.template_id = template_id
        self.ps_replicas = ps_replicas
        self.worker_replicas = worker_replicas
        self.training_step = training_step
        self.interval = interval
        self.batch_size = batch_size
        self.task_id = task_id
        self.tag = tag
        self.rtimes = rtimes
        self.influx_client = influxdb.InfluxDBClient(host='192.168.128.10',port=8086,username='admin',password='admin',database="NODEMESSAGE")
        self.node_list = ['k8s-master','k8s-worker0','k8s-worker2','k8sworker1','k8s-worker3','k8s-worker4','k8s-worker5']
	#self.node_list = ['k8s-master','k8s-worker0','k8s-worker2','k8sworker1']
        self.node_cpu = {}
        self.node_cpu['k8s-master'] = 32000
        self.node_cpu['k8s-worker0'] = 24000
        self.node_cpu['k8s-worker2'] = 24000
        self.node_cpu['k8sworker1'] = 16000
        self.node_cpu['k8s-worker3'] = 24000
        self.node_cpu['k8s-worker4'] = 16000
        self.node_cpu['k8s-worker5'] = 24000
        self.node_memory = {}
        self.node_memory['k8s-master'] = float(251*1024)
        self.node_memory['k8s-worker0'] = float(94*1024)
        self.node_memory['k8s-worker2'] = float(94*1024)
        self.node_memory['k8sworker1'] = float(125*1024)
        self.node_memory['k8s-worker3'] = float(94 * 1024)
        self.node_memory['k8s-worker4'] = float(125 * 1024)
        self.node_memory['k8s-worker5'] = float(94 * 1024)
        self.args = ['--training_step='+str(self.training_step),'--batch_size='+str(self.batch_size),'--interval='+str(self.interval),'--task_id='+str(self.task_id),'--rtimes='+str(self.rtimes),"--tag="+self.tag]

class VGGTask(SubTask):
    def __init__(self,v1,template_id,ps_replicas,worker_replicas,training_step,batch_size,interval,task_id,rtimes,tag,channel1,channel2,channel3,channel4,channel5,num_layer1,num_layer2,num_layer3,num_layer4,num_layer5):
        SubTask.__init__(self,template_id,ps_replicas,worker_replicas,training_step,batch_size,interval,task_id,rtimes,tag)
        self.channel1 = channel1
        self.channel2 = channel2
        self.channel3 = channel3
        self.channel4 = channel4
        self.channel5 = channel5
        self.num_layer1 = num_layer1
        self.num_layer2 = num_layer2
        self.num_layer3 = num_layer3
        self.num_layer4 = num_layer4
        self.num_layer5 = num_layer5
        self.num_layers = num_layer1+num_layer2+num_layer3+num_layer4+num_layer5+3
        self.template = TaskTemplate.VGG
        self.v1 = v1
        self.name = 'vgg-'+str(self.task_id)+'-'+str(self.rtimes)

    def get_node_list(self):
        node_list = [i.metadata.name for i in self.v1.list_node().items]
        return node_list

    def make_args(self):
        self.args.append('--channel1='+str(self.channel1))
        self.args.append('--channel2='+str(self.channel2))
        self.args.append('--channel3='+str(self.channel3))
        self.args.append('--channel4='+str(self.channel4))
        self.args.append('--channel5='+str(self.channel5))
        self.args.append('--num_layer1='+str(self.num_layer1))
        self.args.append('--num_layer2='+str(self.num_layer2))
        self.args.append('--num_layer3='+str(self.num_layer3))
        self.args.append('--num_layer4='+str(self.num_layer4))
        self.args.append('--num_layer5='+str(self.num_layer5))
        self.args.append('--num_layers='+str(self.num_layers))




    def create_tf(self):
        name = 'vgg-'+str(self.task_id)+'-'+str(self.rtimes)
        ns_body = TaskTemplate.NS
        ns_body['metadata']['name'] = name
        if not check_ns(name):
            self.v1.create_namespace(ns_body)
        train_dir = check_path(name)
        time.sleep(12)
        result = self.influx_client.query("select * from "+"NODEMESSAGE"+" group by nodes order by desc limit 3")
        node_list = self.get_node_list()
        result_keys = result.keys()
        nodes = [i[-1]['nodes'] for i in result_keys]
        node_mg = [list(result[i]) for i in result_keys]
        cpu_base = {}
        memory_base = {}
        point_base = {}
        point_base_list = []
        for i in range(len(node_mg)):
            cpu_base[nodes[i]] = 0
            memory_base[nodes[i]] = 0
            point_base[nodes[i]] = 0.0
            for j in range(len(node_mg[0])):
                cpu_base[nodes[i]] += node_mg[i][j]['cpu']
                memory_base[nodes[i]] += node_mg[i][j]['memory']
            cpu_base[nodes[i]] = (cpu_base[nodes[i]] / len(node_mg[0]))/self.node_cpu[nodes[i]]
            memory_base[nodes[i]] = (memory_base[nodes[i]] / len(node_mg[0])) / self.node_memory[nodes[i]]
            tmp = cpu_base[nodes[i]]*0.6+memory_base[nodes[i]]*0.4
            point_base[nodes[i]] = tmp
            point_base_list.append(tmp)

        list.sort(point_base_list)

        for key in nodes:
            command = 'kubectl label nodes '+key+' woksch-'
            os.system(command)
            command2 = 'kubectl label nodes '+key+' wokpro-'
            os.system(command2)
            nod_prori = point_base_list.index(point_base[key])
            priori = ' wokpro=%d' % nod_prori
            command3 = 'kubectl label nodes '+key+priori
            os.system(command3)
            if cpu_base[key] <= 0.57 and memory_base[key] <= 0.6:
                command = 'kubectl label nodes '+key+' woksch=true'
                os.system(command)
            else:
                command = 'kubectl label nodes ' + key + ' woksch=false'
                os.system(command)


        self.template['metadata']['name'] = name
        self.template['metadata']['namespace'] = name
        self.template['spec']['tfReplicaSpecs']['PS']['replicas'] = self.ps_replicas
        self.template['spec']['tfReplicaSpecs']['Worker']['replicas'] = self.worker_replicas
        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['volumes'][0]['name'] = name
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['volumes'][0]['name'] = name
        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['volumes'][0]['hostPath']['path'] = train_dir
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['volumes'][0]['hostPath']['path'] = train_dir

        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['containers'][0]['volumeMounts'][0]['name'] = name
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['containers'][0]['volumeMounts'][0]['name'] = name
        self.make_args()
        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['containers'][0]['args'] = self.args[:]
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['containers'][0]['args'] = self.args[:]

        log_dir = '/tfdata/tfcnn/expjob/'
        # f = open(log_dir+str(name)+'.yaml', "w")
        f = open(log_dir + str(name) + '.yaml', "w")
        yaml.dump(self.template, f)
        f.close()
        response = os.system('kubectl create -f '+log_dir+str(name)+'.yaml')
        if response == 0:
            print('create task sucess')
        else:
            print("Error code:"+str(response))

    def delete_tf(self):
        name = 'vgg-'+str(self.task_id)+'-'+str(self.rtimes)
        log_dir = '/tfdata/tfcnn/expjob/'

        response = os.system('kubectl delete -f ' + log_dir + str(name) + '.yaml')
        if response == 0:
            print('delete task sucess')
        else:
            print("Error code:" + str(response))

        self.v1.delete_namespace(name=name)

class RESTask(SubTask):
    def __init__(self,v1,template_id,ps_replicas,worker_replicas,training_step,batch_size,interval,task_id,rtimes,tag,bottle,layer1,layer2,layer3,layer4,channel1,channel2,channel3,channel4):
        SubTask.__init__(self,template_id,ps_replicas,worker_replicas,training_step,batch_size,interval,task_id,rtimes,tag)
        self.channel1 = channel1
        self.channel2 = channel2
        self.channel3 = channel3
        self.channel4 = channel4
        self.bottle = bottle
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.name = 'res-'+str(self.task_id)+'-'+str(self.rtimes)
        if self.bottle == 1:
            self.num_layers = 3*(layer1+layer4+layer3+layer2)+2
        else:
            self.num_layers = 2 * (layer1 + layer4 + layer3 + layer2) + 2
        self.template = TaskTemplate.RES
        self.v1 = v1

    def get_node_list(self):
        node_list = [i.metadata.name for i in self.v1.list_node().items]
        return node_list

    def make_args(self):
        self.args.append('--bottle=' + str(self.bottle))
        self.args.append('--channel1='+str(self.channel1))
        self.args.append('--channel2='+str(self.channel2))
        self.args.append('--channel3='+str(self.channel3))
        self.args.append('--channel4='+str(self.channel4))
        self.args.append('--layer1='+str(self.layer1))
        self.args.append('--layer2='+str(self.layer2))
        self.args.append('--layer3='+str(self.layer3))
        self.args.append('--layer4='+str(self.layer4))


    def create_tf(self):
        name = 'res-'+str(self.task_id)+'-'+str(self.rtimes)
        ns_body = TaskTemplate.NS
        ns_body['metadata']['name'] = name
        if not check_ns(name):
            self.v1.create_namespace(ns_body)
        train_dir = check_path(name)

        time.sleep(12)
        result = self.influx_client.query("select * from " + "NODEMESSAGE" + " group by nodes order by desc limit 3")
        node_list = self.get_node_list()
        result_keys = result.keys()
        nodes = [i[-1]['nodes'] for i in result_keys]
        node_mg = [list(result[i]) for i in result_keys]
        cpu_base = {}
        memory_base = {}
        point_base = {}
        point_base_list = []
        for i in range(len(node_mg)):
            cpu_base[nodes[i]] = 0
            memory_base[nodes[i]] = 0
            point_base[nodes[i]] = 0.0
            for j in range(len(node_mg[0])):
                cpu_base[nodes[i]] += node_mg[i][j]['cpu']
                memory_base[nodes[i]] += node_mg[i][j]['memory']
            cpu_base[nodes[i]] = (cpu_base[nodes[i]] / len(node_mg[0])) / self.node_cpu[nodes[i]]
            memory_base[nodes[i]] = (memory_base[nodes[i]] / len(node_mg[0])) / self.node_memory[nodes[i]]
            tmp = cpu_base[nodes[i]] * 0.6 + memory_base[nodes[i]] * 0.4
            point_base[nodes[i]] = tmp
            point_base_list.append(tmp)

        list.sort(point_base_list)

        for key in nodes:
            command = 'kubectl label nodes ' + key + ' woksch-'
            os.system(command)
            command2 = 'kubectl label nodes ' + key + ' wokpro-'
            os.system(command2)
            nod_prori = point_base_list.index(point_base[key])
            priori = ' wokpro=%d' % nod_prori
            command3 = 'kubectl label nodes ' + key + priori
            os.system(command3)
            if cpu_base[key] <= 0.6 and memory_base[key] <= 0.6:
                command = 'kubectl label nodes ' + key + ' woksch=true'
                os.system(command)
            else:
                command = 'kubectl label nodes ' + key + ' woksch=false'
                os.system(command)

        self.template['metadata']['name'] = name
        self.template['metadata']['namespace'] = name
        self.template['spec']['tfReplicaSpecs']['PS']['replicas'] = self.ps_replicas
        self.template['spec']['tfReplicaSpecs']['Worker']['replicas'] = self.worker_replicas
        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['volumes'][0]['name'] = name
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['volumes'][0]['name'] = name
        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['volumes'][0]['hostPath']['path'] = train_dir
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['volumes'][0]['hostPath']['path'] = train_dir

        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['containers'][0]['volumeMounts'][0]['name'] = name
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['containers'][0]['volumeMounts'][0]['name'] = name
        self.make_args()
        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['containers'][0]['args'] = self.args[:]
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['containers'][0]['args'] = self.args[:]

        log_dir = '/tfdata/tfcnn/expjob/'
        f = open(log_dir+str(name)+'.yaml', "w")
        yaml.dump(self.template, f)
        f.close()
        response = os.system('kubectl create -f '+log_dir+str(name)+'.yaml')
        if response == 0:
            print('create task sucess')
        else:
            print("Error code:"+str(response))

    def delete_tf(self):
        name = 'res-'+str(self.task_id)+'-'+str(self.rtimes)
        log_dir = '/tfdata/tfcnn/expjob/'

        response = os.system('kubectl delete -f ' + log_dir + str(name) + '.yaml')
        if response == 0:
            print('delete task sucess')
        else:
            print("Error code:" + str(response))

        self.v1.delete_namespace(name=name)

class RETask(SubTask):
    def __init__(self,v1,template_id,ps_replicas,worker_replicas,training_step,batch_size,interval,task_id,rtimes,tag,stack,channel1,channel2,channel3,channel4):
        SubTask.__init__(self,template_id,ps_replicas,worker_replicas,training_step,batch_size,interval,task_id,rtimes,tag)
        self.channel1 = channel1
        self.channel2 = channel2
        self.channel3 = channel3
        self.channel4 = channel4
        self.stack = stack
        self.num_layers = 6*self.stack+2
        self.template = TaskTemplate.RE
        self.name = 're-'+str(self.task_id)+'-'+str(self.rtimes)
        self.v1 = v1

    def get_node_list(self):
        node_list = [i.metadata.name for i in self.v1.list_node().items]
        return node_list

    def make_args(self):
        self.args.append('--stack='+str(self.stack))
        self.args.append('--channel1='+str(self.channel1))
        self.args.append('--channel2='+str(self.channel2))
        self.args.append('--channel3='+str(self.channel3))
        self.args.append('--channel4='+str(self.channel4))

    def create_tf(self):
        name = 're-'+str(self.task_id)+'-'+str(self.rtimes)
        ns_body = TaskTemplate.NS
        ns_body['metadata']['name'] = name
        if not check_ns(name):
            self.v1.create_namespace(ns_body)
        train_dir = check_path(name)

        time.sleep(12)
        result = self.influx_client.query("select * from " + "NODEMESSAGE" + " group by nodes order by desc limit 3")
        node_list = self.get_node_list()
        result_keys = result.keys()
        nodes = [i[-1]['nodes'] for i in result_keys]
        node_mg = [list(result[i]) for i in result_keys]
        cpu_base = {}
        memory_base = {}
        point_base = {}
        point_base_list = []
        for i in range(len(node_mg)):
            cpu_base[nodes[i]] = 0
            memory_base[nodes[i]] = 0
            point_base[nodes[i]] = 0.0
            for j in range(len(node_mg[0])):
                cpu_base[nodes[i]] += node_mg[i][j]['cpu']
                memory_base[nodes[i]] += node_mg[i][j]['memory']
            cpu_base[nodes[i]] = (cpu_base[nodes[i]] / len(node_mg[0])) / self.node_cpu[nodes[i]]
            memory_base[nodes[i]] = (memory_base[nodes[i]] / len(node_mg[0])) / self.node_memory[nodes[i]]
            tmp = cpu_base[nodes[i]] * 0.6 + memory_base[nodes[i]] * 0.4
            point_base[nodes[i]] = tmp
            point_base_list.append(tmp)

        list.sort(point_base_list)

        for key in nodes:
            command = 'kubectl label nodes ' + key + ' woksch-'
            os.system(command)
            command2 = 'kubectl label nodes ' + key + ' wokpro-'
            os.system(command2)
            nod_prori = point_base_list.index(point_base[key])
            priori = ' wokpro=%d' % nod_prori
            command3 = 'kubectl label nodes ' + key + priori
            os.system(command3)
            if cpu_base[key] <= 0.6 and memory_base[key] <= 0.6:
                command = 'kubectl label nodes ' + key + ' woksch=true'
                os.system(command)
            else:
                command = 'kubectl label nodes ' + key + ' woksch=false'
                os.system(command)

        self.template['metadata']['name'] = name
        self.template['metadata']['namespace'] = name
        self.template['spec']['tfReplicaSpecs']['PS']['replicas'] = self.ps_replicas
        self.template['spec']['tfReplicaSpecs']['Worker']['replicas'] = self.worker_replicas
        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['volumes'][0]['name'] = name
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['volumes'][0]['name'] = name
        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['volumes'][0]['hostPath']['path'] = train_dir
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['volumes'][0]['hostPath']['path'] = train_dir

        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['containers'][0]['volumeMounts'][0]['name'] = name
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['containers'][0]['volumeMounts'][0]['name'] = name
        self.make_args()
        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['containers'][0]['args'] = self.args[:]
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['containers'][0]['args'] = self.args[:]

        log_dir = '/tfdata/tfcnn/expjob/'
        f = open(log_dir+str(name)+'.yaml', "w")
        yaml.dump(self.template, f)
        f.close()
        response = os.system('kubectl create -f '+log_dir+str(name)+'.yaml')
        if response == 0:
            print('create task sucess')
        else:
            print("Error code:"+str(response))

    def delete_tf(self):
        name = 're-'+str(self.task_id)+'-'+str(self.rtimes)
        log_dir = '/tfdata/tfcnn/expjob/'

        response = os.system('kubectl delete -f ' + log_dir + str(name) + '.yaml')
        if response == 0:
            print('delete task sucess')
        else:
            print("Error code:" + str(response))

        self.v1.delete_namespace(name=name)


class XCETask(SubTask):
    def __init__(self,v1,template_id,ps_replicas,worker_replicas,training_step,batch_size,interval,task_id,rtimes,tag,repeat,channel1,channel2,channel3,channel4,channel5,channel6,channel7,channel8):
        SubTask.__init__(self,template_id,ps_replicas,worker_replicas,training_step,batch_size,interval,task_id,rtimes,tag)
        self.channel1 = channel1
        self.channel2 = channel2
        self.channel3 = channel3
        self.channel4 = channel4
        self.channel5 = channel5
        self.channel6 = channel6
        self.channel7 = channel7
        self.channel8 = channel8
        self.repeat = repeat
        self.template = TaskTemplate.XCEPTION
        self.v1 = v1
        self.name = 'xception-'+str(self.task_id)+'-'+str(self.rtimes)

    def get_node_list(self):
        node_list = [i.metadata.name for i in self.v1.list_node().items]
        return node_list

    def make_args(self):
        self.args.append('--repeat='+str(self.repeat))
        self.args.append('--channel1='+str(self.channel1))
        self.args.append('--channel2='+str(self.channel2))
        self.args.append('--channel3='+str(self.channel3))
        self.args.append('--channel4='+str(self.channel4))
        self.args.append('--channel5=' + str(self.channel5))
        self.args.append('--channel6=' + str(self.channel6))
        self.args.append('--channel7=' + str(self.channel7))
        self.args.append('--channel8=' + str(self.channel8))

    def create_tf(self):
        name = 'xception-'+str(self.task_id)+'-'+str(self.rtimes)
        ns_body = TaskTemplate.NS
        ns_body['metadata']['name'] = name
        if not check_ns(name):
            self.v1.create_namespace(ns_body)
        train_dir = check_path(name)

        time.sleep(12)
        result = self.influx_client.query("select * from " + "NODEMESSAGE" + " group by nodes order by desc limit 3")
        node_list = self.get_node_list()
        result_keys = result.keys()
        nodes = [i[-1]['nodes'] for i in result_keys]
        node_mg = [list(result[i]) for i in result_keys]
        cpu_base = {}
        memory_base = {}
        point_base = {}
        point_base_list = []
        for i in range(len(node_mg)):
            cpu_base[nodes[i]] = 0
            memory_base[nodes[i]] = 0
            point_base[nodes[i]] = 0.0
            for j in range(len(node_mg[0])):
                cpu_base[nodes[i]] += node_mg[i][j]['cpu']
                memory_base[nodes[i]] += node_mg[i][j]['memory']
            cpu_base[nodes[i]] = (cpu_base[nodes[i]] / len(node_mg[0])) / self.node_cpu[nodes[i]]
            memory_base[nodes[i]] = (memory_base[nodes[i]] / len(node_mg[0])) / self.node_memory[nodes[i]]
            tmp = cpu_base[nodes[i]] * 0.6 + memory_base[nodes[i]] * 0.4
            point_base[nodes[i]] = tmp
            point_base_list.append(tmp)

        list.sort(point_base_list)

        for key in nodes:
            command = 'kubectl label nodes ' + key + ' woksch-'
            os.system(command)
            command2 = 'kubectl label nodes ' + key + ' wokpro-'
            os.system(command2)
            nod_prori = point_base_list.index(point_base[key])
            priori = ' wokpro=%d' % nod_prori
            command3 = 'kubectl label nodes ' + key + priori
            os.system(command3)
            if cpu_base[key] <= 0.6 and memory_base[key] <= 0.6:
                command = 'kubectl label nodes ' + key + ' woksch=true'
                os.system(command)
            else:
                command = 'kubectl label nodes ' + key + ' woksch=false'
                os.system(command)

        self.template['metadata']['name'] = name
        self.template['metadata']['namespace'] = name
        self.template['spec']['tfReplicaSpecs']['PS']['replicas'] = self.ps_replicas
        self.template['spec']['tfReplicaSpecs']['Worker']['replicas'] = self.worker_replicas
        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['volumes'][0]['name'] = name
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['volumes'][0]['name'] = name
        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['volumes'][0]['hostPath']['path'] = train_dir
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['volumes'][0]['hostPath']['path'] = train_dir

        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['containers'][0]['volumeMounts'][0]['name'] = name
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['containers'][0]['volumeMounts'][0]['name'] = name
        self.make_args()
        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['containers'][0]['args'] = self.args[:]
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['containers'][0]['args'] = self.args[:]

        log_dir = '/tfdata/tfcnn/expjob/'
        f = open(log_dir+str(name)+'.yaml', "w")
        yaml.dump(self.template, f)
        f.close()
        response = os.system('kubectl create -f '+log_dir+str(name)+'.yaml')
        if response == 0:
            print('create task sucess')
        else:
            print("Error code:"+str(response))

    def delete_tf(self):
        name = 'xception-'+str(self.task_id)+'-'+str(self.rtimes)
        log_dir = '/tfdata/tfcnn/expjob/'

        response = os.system('kubectl delete -f ' + log_dir + str(name) + '.yaml')
        if response == 0:
            print('delete task sucess')
        else:
            print("Error code:" + str(response))

        self.v1.delete_namespace(name=name)


class DENTask(SubTask):
    def __init__(self,v1,template_id,ps_replicas,worker_replicas,training_step,batch_size,interval,task_id,rtimes,tag,L,k,BC):
        SubTask.__init__(self,template_id,ps_replicas,worker_replicas,training_step,batch_size,interval,task_id,rtimes,tag)
        self.L = L
        self.k = k
        self.BC = BC
        self.template = TaskTemplate.DEN
        self.v1 = v1
        self.name = 'den-'+str(self.task_id)+'-'+str(self.rtimes)

    def get_node_list(self):
        node_list = [i.metadata.name for i in self.v1.list_node().items]
        return node_list

    def make_args(self):
        self.args.append('--L='+str(self.L))
        self.args.append('--k='+str(self.k))
        self.args.append('--BC='+str(self.BC))

    def create_tf(self):
        name = 'den-'+str(self.task_id)+'-'+str(self.rtimes)
        ns_body = TaskTemplate.NS
        ns_body['metadata']['name'] = name
        if not check_ns(name):
            self.v1.create_namespace(ns_body)
        train_dir = check_path(name)

        time.sleep(12)
        result = self.influx_client.query("select * from " + "NODEMESSAGE" + " group by nodes order by desc limit 3")
        node_list = self.get_node_list()
        result_keys = result.keys()
        nodes = [i[-1]['nodes'] for i in result_keys]
        node_mg = [list(result[i]) for i in result_keys]
        cpu_base = {}
        memory_base = {}
        point_base = {}
        point_base_list = []
        for i in range(len(node_mg)):
            cpu_base[nodes[i]] = 0
            memory_base[nodes[i]] = 0
            point_base[nodes[i]] = 0.0
            for j in range(len(node_mg[0])):
                cpu_base[nodes[i]] += node_mg[i][j]['cpu']
                memory_base[nodes[i]] += node_mg[i][j]['memory']
            cpu_base[nodes[i]] = (cpu_base[nodes[i]] / len(node_mg[0])) / self.node_cpu[nodes[i]]
            memory_base[nodes[i]] = (memory_base[nodes[i]] / len(node_mg[0])) / self.node_memory[nodes[i]]
            tmp = cpu_base[nodes[i]] * 0.6 + memory_base[nodes[i]] * 0.4
            point_base[nodes[i]] = tmp
            point_base_list.append(tmp)

        list.sort(point_base_list)

        for key in nodes:
            command = 'kubectl label nodes ' + key + ' woksch-'
            os.system(command)
            command2 = 'kubectl label nodes ' + key + ' wokpro-'
            os.system(command2)
            nod_prori = point_base_list.index(point_base[key])
            priori = ' wokpro=%d' % nod_prori
            command3 = 'kubectl label nodes ' + key + priori
            os.system(command3)
            if cpu_base[key] <= 0.6 and memory_base[key] <= 0.6:
                command = 'kubectl label nodes ' + key + ' woksch=true'
                os.system(command)
            else:
                command = 'kubectl label nodes ' + key + ' woksch=false'
                os.system(command)

        self.template['metadata']['name'] = name
        self.template['metadata']['namespace'] = name
        self.template['spec']['tfReplicaSpecs']['PS']['replicas'] = self.ps_replicas
        self.template['spec']['tfReplicaSpecs']['Worker']['replicas'] = self.worker_replicas
        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['volumes'][0]['name'] = name
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['volumes'][0]['name'] = name
        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['volumes'][0]['hostPath']['path'] = train_dir
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['volumes'][0]['hostPath']['path'] = train_dir

        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['containers'][0]['volumeMounts'][0]['name'] = name
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['containers'][0]['volumeMounts'][0]['name'] = name
        self.make_args()
        self.template['spec']['tfReplicaSpecs']['PS']['template']['spec']['containers'][0]['args'] = self.args[:]
        self.template['spec']['tfReplicaSpecs']['Worker']['template']['spec']['containers'][0]['args'] = self.args[:]

        log_dir = '/tfdata/tfcnn/expjob/'
        f = open(log_dir+str(name)+'.yaml', "w")
        yaml.dump(self.template, f)
        f.close()
        response = os.system('kubectl create -f '+log_dir+str(name)+'.yaml')
        if response == 0:
            print('create task sucess')
        else:
            print("Error code:"+str(response))

    def delete_tf(self):
        name = 'den-'+str(self.task_id)+'-'+str(self.rtimes)
        log_dir = '/tfdata/tfcnn/expjob/'

        response = os.system('kubectl delete -f ' + log_dir + str(name) + '.yaml')
        if response == 0:
            print('delete task sucess')
        else:
            print("Error code:" + str(response))

        self.v1.delete_namespace(name=name)


if __name__ == '__main__':
    kubernetes.config.load_kube_config()
    v1 = kubernetes.client.CoreV1Api()
    # v1.create_namespace()
    v1.list_namespace()
    check_path('ceshi')
    # vgg = VGGTask(1,2,4,80,1.0,2,1,"ms",32,64,128,256,512,2,3,3,4,4)
    # vgg.create_tf()
