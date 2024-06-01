# -*- coding: UTF-8 -*- 

import datetime
import json

from django.contrib.auth.hashers import check_password, make_password
from django.core import serializers
from django.db import connection
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from cmdb.models import host, hostUser, dbGroup, dbInstance
from utils.jsonExt import DateEncoder
from utils.logUtil import getLogger


# from cmdb.models import dbCluster
logger = getLogger()

@csrf_exempt    
def addChangeHostInfo(request):
    '''
    新增主机
    修改主机
    '''
    v_hostId = request.POST.get('host_id')
    v_businessName = request.POST.get('business_name')
    v_serviceEnv = request.POST.get('service_env')
    v_hostName = request.POST.get('host_name')
    v_intranetIpAddr = request.POST.get('intranet_ipaddr')
    v_publicIpAddr = request.POST.get('public_ipaddr')
    v_sshPort = request.POST.get('ssh_port')
    v_hostType = request.POST.get('host_type')
    v_hostRole = request.POST.get('host_role')
    v_hostDesc = request.POST.get('host_desc')
    
    print(v_hostId, v_businessName, v_serviceEnv, v_hostName, v_intranetIpAddr, v_publicIpAddr, v_sshPort, v_hostType, v_hostRole, v_hostDesc)
        
    if v_hostId == '' or v_hostId is None:
        # 新增
        try:   
            hostObj = host(businessName=v_businessName, serviceEnv=v_serviceEnv, hostName=v_hostName, intranetIpAddr=v_intranetIpAddr, publicIpAddr=v_publicIpAddr, sshPort=v_sshPort, hostType=v_hostType, hostRole=v_hostRole, hostDesc=v_hostDesc)
            hostObj.save()
            result = {'status':1, 'msg':'保存成功!', 'data':''}
            return HttpResponse(json.dumps(result), content_type='application/json')
        except Exception as e:
            result = {'status':2, 'msg':'保存失败!'+str(e), 'data':''}
            return HttpResponse(json.dumps(result), content_type='application/json')        
    else:
        # 修改
        try:
            hostObj = host.objects.filter(id=v_hostId)
            hostObj.update(businessName=v_businessName, serviceEnv=v_serviceEnv, hostName=v_hostName, intranetIpAddr=v_intranetIpAddr, publicIpAddr=v_publicIpAddr, sshPort=v_sshPort, hostType=v_hostType, hostRole=v_hostRole, hostDesc=v_hostDesc)
#             masterConfigObj.save()
            result = {'status':1, 'msg':'修改成功!', 'data':''}
            return HttpResponse(json.dumps(result), content_type='application/json')
        except Exception as e:
            result = {'status':2, 'msg':'修改失败!'+str(e), 'data':''}
            return HttpResponse(json.dumps(result), content_type='application/json')      


@csrf_exempt
def getHostDetailInfo(request):
    hostId = request.POST['hostId']
    
    try:
        hostObj = host.objects.get(id=hostId)
        hostJson = hostObj.toJSON()
        
        result = {'status':1, 'msg':'请求成功', 'obj':hostJson}
        print(result)
        return HttpResponse(json.dumps(result), content_type='application/json')
    except Exception as e:
        print(e)
        result = {'status':2, 'msg':'请求失败!'+str(e), 'data':''}
        return HttpResponse(json.dumps(result), content_type='application/json')
 
@csrf_exempt    
def delHost(request):
    hostId = request.POST['hostId']

    if hostId == "" or hostId is None:
        result = {'status':3, 'msg':'未选中任何记录!', 'data':''}
        return HttpResponse(json.dumps(result), content_type='application/json')
    else:    
        try:
            delResult = host.objects.filter(id=hostId).delete()
            print(delResult)
            result = {'status':1, 'msg':'删除成功!', 'data':delResult}
            return HttpResponse(json.dumps(result), content_type='application/json')            
        except Exception as e:
            print(e)
            result = {'status':2, 'msg':'删除失败!'+str(e), 'data':''}
            return HttpResponse(json.dumps(result), content_type='application/json')
    
@csrf_exempt    
def addChangeHostUserInfo(request):
    '''
    新增主机用户
    修改主机用户
    '''
    v_hostUserId = request.POST.get('host_user_id')
    v_hostId = request.POST.get('host_id')
    v_hostUser = request.POST.get('host_user')
    v_hostPasswd = request.POST.get('host_passwd')
    v_userDesc = request.POST.get('user_desc')
    
    print(v_hostUserId, v_hostId, v_hostUser, v_hostPasswd, v_userDesc)
        
    if v_hostUserId == '' or v_hostUserId is None:
        # 新增
        try: 
            hostObj = host.objects.get(id=v_hostId)       
            hostUserObj = hostUser(hostUser=v_hostUser, hostPasswd=v_hostPasswd, userDesc=v_userDesc, host=hostObj)
            hostUserObj.save()
            result = {'status':1, 'msg':'保存成功!', 'data':''}
            return HttpResponse(json.dumps(result), content_type='application/json')
        except Exception as e:
            logger.error(str(e))
            result = {'status':2, 'msg':'保存失败!', 'data':''}
            return HttpResponse(json.dumps(result), content_type='application/json')        
    else:
        # 修改
        try:
            hostUserObj = hostUser.objects.filter(id=v_hostUserId)
            hostUserObj.update(hostUser=v_hostUser, hostPasswd=v_hostPasswd, userDesc=v_userDesc)
#             masterConfigObj.save()
            result = {'status':1, 'msg':'修改成功!', 'data':''}
            return HttpResponse(json.dumps(result), content_type='application/json')
        except Exception as e:
            logger.error(str(e))
            result = {'status':2, 'msg':'修改失败!', 'data':''}
            return HttpResponse(json.dumps(result), content_type='application/json')
       
@csrf_exempt
def getHostUserDetailInfo(request):
    hostUserId = request.POST['hostUserId'].strip()
    
    try:
        hostUserInfo = hostUser.objects.filter(id=hostUserId)
        hostUserJson = serializers.serialize("json", hostUserInfo, use_natural_foreign_keys=True)
        result = {'status':1, 'msg':'请求成功', 'hostUserJson':hostUserJson}
        print(result)
        return HttpResponse(json.dumps(result), content_type='application/json')
    except Exception as e:
        print(e)
        result = {'status':2, 'msg':'请求失败!'+str(e), 'data':''}
        return HttpResponse(json.dumps(result), content_type='application/json')
  
@csrf_exempt    
def delHostUser(request):
    hostUserId = request.POST['hostUserId']

    if hostUserId == "" or hostUserId is None:
        result = {'status':3, 'msg':'未选中任何记录!', 'data':''}
        return HttpResponse(json.dumps(result), content_type='application/json')
    else:    
        try:
            delResult = hostUser.objects.filter(id=hostUserId).delete()
            print(delResult)
            logger.error(delResult)
            result = {'status':1, 'msg':'删除成功!', 'data':delResult}
            return HttpResponse(json.dumps(result), content_type='application/json')            
        except Exception as e:
            print(e)
            logger.error(e)
            result = {'status':2, 'msg':'删除失败!', 'data':''}
            return HttpResponse(json.dumps(result), content_type='application/json')        

@csrf_exempt    
def addChangeDbGroupInfo(request):
    '''
    新增数据库组
    修改数据库组
    '''
                    
    v_groupId = request.POST.get('group_id')
    v_businessName = request.POST.get('business_name')
    v_groupName = request.POST.get('group_name')
    v_groupStatus = request.POST.get('group_status')
    v_groupDesc = request.POST.get('group_desc')
    v_groupEnv = request.POST.get('group_env')
    
    
    print(v_groupId, v_businessName, v_groupName, v_groupEnv, v_groupStatus, v_groupDesc)
    logger.info("保存或修改数据库组信息，接收前端参数：", v_groupId, v_businessName, v_groupName, v_groupEnv, v_groupStatus, v_groupDesc)
        
    if v_groupId == '' or v_groupId is None:
        # 新增
        try:     
            dbGroupObj = dbGroup(businessName=v_businessName, groupName=v_groupName, groupEnv=v_groupEnv, groupStatus=v_groupStatus, groupDesc=v_groupDesc)
            dbGroupObj.save()
            result = {'status':1, 'msg':'保存成功!', 'data':''}
            return HttpResponse(json.dumps(result), content_type='application/json')
        except Exception as e:
            logger.error(str(e))
            result = {'status':2, 'msg':'保存失败!', 'data':''}
            return HttpResponse(json.dumps(result), content_type='application/json')        
    else:
        # 修改
        try:
            dbGroupObj = dbGroup.objects.filter(id=v_groupId)
            dbGroupObj.update(businessName=v_businessName, groupName=v_groupName, groupEnv=v_groupEnv, groupStatus=v_groupStatus, groupDesc=v_groupDesc)
#             masterConfigObj.save()
            result = {'status':1, 'msg':'修改成功!', 'data':''}
            return HttpResponse(json.dumps(result), content_type='application/json')
        except Exception as e:
            logger.error(str(e))
            result = {'status':2, 'msg':'修改失败!', 'data':''}
            return HttpResponse(json.dumps(result), content_type='application/json')

# @csrf_exempt
# def getDbClusterDetailInfo(request):
#     clusterId = request.POST['clusterId']
#     
#     try:
#         dbClusterObj = dbCluster.objects.get(id=clusterId)
#         dbClusterJson = dbClusterObj.toJSON()
#         
#         result = {'status':1, 'msg':'请求成功', 'obj':dbClusterJson}
#         print(result)
#         return HttpResponse(json.dumps(result), content_type='application/json')
#     except Exception as e:
#         print(e)
#         result = {'status':2, 'msg':'请求失败!'+str(e), 'data':''}
#         return HttpResponse(json.dumps(result), content_type='application/json')
    
@csrf_exempt
def getDbGroupDetailInfo(request):
    groupId = request.POST['groupId']
    
    try:
        dbGroupObj = dbGroup.objects.get(id=groupId)
        dbGroupJson = dbGroupObj.toJSON()
        
        result = {'status':1, 'msg':'请求成功', 'obj':dbGroupJson}
        print(result)
        return HttpResponse(json.dumps(result), content_type='application/json')
    except Exception as e:
        print(e)
        result = {'status':2, 'msg':'请求失败!'+str(e), 'data':''}
        return HttpResponse(json.dumps(result), content_type='application/json')
            

@csrf_exempt    
def addChangeDbInstanceInfo(request):
    '''
            新增数据库实例
            修改数据库实例
    '''
    v_instanceId = request.POST.get('instance_id')                          
    v_groupId = request.POST.get('group_id')
    v_host_id = request.POST.get('host_id')
    v_instanceName = request.POST.get('instance_env')
    v_instanceType = request.POST.get('instance_type')
    v_portNum = request.POST.get('port_num')
    v_instanceRole = request.POST.get('instance_role')
    v_instanceStatus = request.POST.get('instance_status')
    v_instanceDesc = request.POST.get('instance_desc')
    
    print(v_instanceId, v_groupId, v_host_id, v_instanceName, v_instanceType, v_portNum, v_instanceRole, v_instanceStatus, v_instanceDesc)
    logger.info("保存或修改数据库实例信息，接收前端参数：", v_instanceId, v_groupId, v_host_id, v_instanceName, v_instanceType, v_portNum, v_instanceRole, v_instanceStatus, v_instanceDesc)
        
    if v_instanceId == '' or v_instanceId is None:
        # 新增
        try:
            dbGroupObj = dbGroup.objects.get(id=v_groupId)
            hostObj = host.objects.get(id=v_host_id)
            print(hostObj)
            dbInstanceObj = dbInstance(groupName=dbGroupObj, host=hostObj, instanceName=v_instanceName, instanceType=v_instanceType, portNum=v_portNum, instanceRole=v_instanceRole, instanceStatus=v_instanceStatus, instanceDesc=v_instanceDesc)
            dbInstanceObj.save()
            result = {'status':1, 'msg':'保存成功!', 'data':''}
            return HttpResponse(json.dumps(result), content_type='application/json')
        except Exception as e:
            print(e)
            logger.error(str(e))
            result = {'status':2, 'msg':'保存失败!', 'data':''}
            return HttpResponse(json.dumps(result), content_type='application/json')        
    else:
        # 修改
        try:
            dbGroupObj = dbGroup.objects.get(id=v_groupId)
            hostObj = host.objects.get(id=v_host_id)
            dbInstanceObj = dbInstance.objects.filter(id=v_instanceId)
            dbInstanceObj.update(groupName=dbGroupObj, host=hostObj, instanceName=v_instanceName, instanceType=v_instanceType, portNum=v_portNum, instanceRole=v_instanceRole, instanceStatus=v_instanceStatus, instanceDesc=v_instanceDesc)
#             masterConfigObj.save()
            result = {'status':1, 'msg':'修改成功!', 'data':''}
            return HttpResponse(json.dumps(result), content_type='application/json')
        except Exception as e:
            logger.error(str(e))
            result = {'status':2, 'msg':'修改失败!', 'data':''}
            return HttpResponse(json.dumps(result), content_type='application/json')
        
@csrf_exempt
def getDbInstanceDetailInfo(request):
    instanceId = request.POST['instanceId'].strip()
    
    try:
        dbInstanceInfo = dbInstance.objects.filter(id=instanceId)
        dbInstanceJson = serializers.serialize("json", dbInstanceInfo, use_natural_foreign_keys=True)
        result = {'status':1, 'msg':'请求成功', 'dbInstanceJson':dbInstanceJson}
        print(result)
        return HttpResponse(json.dumps(result), content_type='application/json')
    except Exception as e:
        print(e)
        result = {'status':2, 'msg':'请求失败!'+str(e), 'data':''}
        return HttpResponse(json.dumps(result), content_type='application/json')
          
#     conn = connection.cursor()
#     try:
#         conn.execute('SELECT cdi.*, ch.host_name, ch.intranet_ip_addr, cdg.group_name FROM cmdb_db_instance cdi inner join cmdb_host ch on cdi.host = ch.id inner join cmdb_db_group cdg on cdi.db_group = cdg.id WHERE cdi.id = %s', [instanceId])
#         dbInstanceInfo = conn.fetchall()
#         print(dbInstanceInfo)
#         dbInstanceJson = serializers.serialize("json", dbInstanceInfo)
#         result = {'status':1, 'msg':'请求成功', 'dbInstanceInfo':dbInstanceInfo}
#         print(result)
#         return HttpResponse(json.dumps(result, cls=DateEncoder), content_type='application/json')        
#     except Exception as e:
#         print(e)
#         result = {'status':2, 'msg':'请求失败!'+str(e), 'data':''}
#         return HttpResponse(json.dumps(result), content_type='application/json')
#     finally:
#         conn.close()    
    
#     try:    
#         dbInstanceInfo = dbInstance.objects.raw('SELECT * FROM cmdb_db_instance WHERE id = %d', [instanceId])
#         dbInstanceJson = serializers.serialize("json", dbInstanceInfo)
#          
#         print(dbInstanceJson[0].fields.host)
#         print(type(dbInstanceJson[0].fields.host))
#          
#         hostInfo = host.objects.raw('SELECT * FROM cmdb_host WHERE id = %d', [int(dbInstanceJson[0].fields.host)])
#         hostJson = serializers.serialize("json", hostInfo)
#         print(hostJson)
#  
#         result = {'status':1, 'msg':'请求成功', 'dbInstanceJson':dbInstanceJson}
#         print(result)
#         return HttpResponse(json.dumps(result), content_type='application/json')
#     except Exception as e:
#         print(e)
#         result = {'status':2, 'msg':'请求失败!'+str(e), 'data':''}
#         return HttpResponse(json.dumps(result), content_type='application/json')


@csrf_exempt    
def delDbInstance(request):
    instanceId = request.POST['instanceId']

    if instanceId == "" or instanceId is None:
        result = {'status':3, 'msg':'未选中任何记录!', 'data':''}
        return HttpResponse(json.dumps(result), content_type='application/json')
    else:    
        try:
            delResult = dbInstance.objects.filter(id=instanceId).delete()
            print(delResult)
            logger.error(delResult)
            result = {'status':1, 'msg':'删除成功!', 'data':delResult}
            return HttpResponse(json.dumps(result), content_type='application/json')            
        except Exception as e:
            print(e)
            logger.error(e)
            result = {'status':2, 'msg':'删除失败!', 'data':''}
            return HttpResponse(json.dumps(result), content_type='application/json')        

        
# @csrf_exempt    
# def addChangeDbClusterInfo(request):
#     '''
#             新增集群信息
#             修改集群信息
#     '''
#     v_clusterId = request.POST.get('cluster_id')
#     v_clusterName = request.POST.get('cluster_name')
#     v_clusterStatus = request.POST.get('cluster_status')
#     v_clusterDesc = request.POST.get('cluster_desc')
#             
#     print("begin add Cluster: ", v_clusterId, v_clusterName, v_clusterStatus, v_clusterDesc)
#         
#     if v_clusterId == '' or v_clusterId is None:
#         # 新增
#         try:     
#             dbClusterObj = dbCluster(clusterName=v_clusterName, clusterStatus=v_clusterStatus, clusterDesc=v_clusterDesc)
#             dbClusterObj.save()
#             result = {'status':1, 'msg':'保存成功!', 'data':''}
#             return HttpResponse(json.dumps(result), content_type='application/json')
#         except Exception as e:
#             logger.error(str(e))
#             result = {'status':2, 'msg':'保存失败!', 'data':''}
#             return HttpResponse(json.dumps(result), content_type='application/json')        
#     else:
#         # 修改
#         try:
#             dbClusterObj = dbCluster.objects.filter(id=v_clusterId)
#             dbClusterObj.update(clusterName=v_clusterName, clusterStatus=v_clusterStatus, clusterDesc=v_clusterDesc)
# #             masterConfigObj.save()
#             result = {'status':1, 'msg':'修改成功!', 'data':''}
#             return HttpResponse(json.dumps(result), content_type='application/json')
#         except Exception as e:
#             logger.error(str(e))
#             result = {'status':2, 'msg':'修改失败!', 'data':''}
#             return HttpResponse(json.dumps(result), content_type='application/json')
#         
# @csrf_exempt    
# def delDbCluster(request):
#     v_clusterId = request.POST['cluster_id']
# 
#     if v_clusterId == "" or v_clusterId is None:
#         result = {'status':3, 'msg':'未选中任何记录!', 'data':''}
#         return HttpResponse(json.dumps(result), content_type='application/json')
#     else:    
#         try:
#             delResult = dbCluster.objects.filter(id=v_clusterId).delete()
#             print(delResult)
#             logger.info(delResult)
#             result = {'status':1, 'msg':'删除成功!', 'data':delResult}
#             return HttpResponse(json.dumps(result), content_type='application/json')            
#         except Exception as e:
#             print(e)
#             logger.error(e)
#             result = {'status':2, 'msg':'删除失败!', 'data':''}
#             return HttpResponse(json.dumps(result), content_type='application/json')              
        