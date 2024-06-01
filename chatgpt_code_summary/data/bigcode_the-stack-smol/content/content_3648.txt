import os
class ConfigParams:
    def __init__(self,configPath):
        self.env_dist = os.environ
        #权限验证
        self.api_key = ""

        # userID = ""
        # ip = "0.0.0.0"

        #模型相关存放根目录
        self.modelPath = os.path.join(os.getcwd(),"model")

        cpuCores = 0
        threads = 2
        port = 33388
        batchSize = 10
        #每个算法使用的GPU数量
        self.GPUDevices = 1

        topK = 80
        featureSize = 512

        zmqthreads = 2

        self.CPU = 0
        self.zmqAddr = "tcp://{}:5560".format(self.env_dist["ZMQ_ADDR"]) if "ZMQ_ADDR" in self.env_dist else "tcp://127.0.0.1:5570"
        print(str(self.zmqAddr))


        self.helmet_ids = self.parseAI("HELMET") if "HELMET" in self.env_dist else []
        self.pose_ids = self.parseAI("POSE") if "POSE" in self.env_dist else []
        self.track_coal_ids = self.parseAI("TRACK_COAL") if "TRACK_COAL" in self.env_dist else []
        self.smoke_phone_ids = self.parseAI("SMOKEPHONE") if "SMOKEPHONE" in self.env_dist else []


        # self.helmet_ids = [1,1,1]

        # self.pose_ids = []
        # self.track_coal_ids = []
        # self.smoke_phone_ids = []


    def loadConfig(self,configPath):
        pass
    def generateDefaultConfig(self,configPath):
        pass

    def initEasylogging(self,logConfig):
        pass
    def printParams(self):
        print("run configParams function printParams")
        pass
    def parseAI(self,key):
        ai_ids = []
        for i in self.env_dist[key].split(','):
            ai_ids.append(int(i))

        return ai_ids
