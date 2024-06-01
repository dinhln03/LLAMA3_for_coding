#coding=utf-8
#author@alingse
#2016.06.21


hdfs_schema = 'hdfs://'
file_schema = 'file://'


class hdfsCluster(object):
    """ 一个hdfs 资源 hdfs uri,path,账户密码认证
    """
    def __init__(self,host,port=9000,schema=hdfs_schema):
        """ 目前只需要host和port """
        self.host = host
        self.port = port
        self.schema = schema
        self._path = '/'
        self._status = None
    
    @property
    def status(self):
        return self._status
    
    @status.setter
    def status(self,value):
        if value in [None,True,False]:
            self._status = value
    
    @property
    def path(self):
        return self._path

    @path.setter
    def path(self,value):
        if value.startswith('/') and value.endswith('/'):
            self._path = value
            self._status = None

    @property
    def uri_head(self):
        """ 返回 uri 的 head"""
        head = self.schema + '{}:{}'.format(self.host,self.port)
        return head

    @property
    def uri(self):
        """ 返回当前路径"""
        _uri = self.schema + '{}:{}{}'.format(self.host,self.port,self._path)
        return _uri



if __name__ == '__main__':
    hdfs = hdfsCluster('localhost','9000')
    hdfs.path = '/hive/'
    print(hdfs.uri)
    print(hdfs.uri_head)
    
    
