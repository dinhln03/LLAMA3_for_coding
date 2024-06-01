#!/bin/env python 
#
#  Copyright 2014 Alcatel-Lucent Enterprise.
#
#  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
#  except in compliance with the License. You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software distributed under the License
#  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
#  either express or implied. See the License for the specific language governing permissions
#  and limitations under the License.
#
#
# $File: omniswitch_restful_driver.py$

# $Build: OONP_H_R01_6$

# $Date: 05/06/2014 12:10:39$

# $Author: vapoonat$

#
#

import logging
import os
import sys
import importlib
import traceback
import urllib
import urllib2
import time

import thread
import threading

from neutron.plugins.omniswitch.consumer import AOSAPI, AOSConnection
#from neutron.plugins.omniswitch.omniswitch_driver_base import OmniSwitchDeviceDriverBase

LOG = logging.getLogger(__name__)


class OmniSwitchRestfulDriver(object):

    """ 
    Name:        OmniSwitchRestfulDriver 
    Description: OmniSwitch device driver to communicate with OS6900 and OS10K devices which support
                 RESTful interface.

    Details:     It is used by OmniSwitchDevicePluginV2 to perform the necessary configuration on the physical 
                 switches as response to OpenStack networking APIs. This driver is used only for OS6900, OS10K 
                 and OS6860 devices which support RESTful APIs to configure them. This driver requires the following 
                 minimum version of AOS SW to be running on these devices...
                        OS10K  : 732-R01-GA
                        OS6900 : 733-R01-GA
                        OS6860 : 811-R01--GA

                 It uses the "consumer.py" library provided as a reference implementation from the AOS/OmniSwitch
                 point of view. No changes is made as part of OmniSwitch plug-in or driver development. For 
                 latest version of consumer.py, refer "//depot/7.3.3.R01/sw/management/web/consumer/consumer.py".
                 For any issues/bugs with the library, please contact AOS 7x WebService module owner
                 (Chris Ravanscoft)
    """
    switch_ip = None
    switch_login = None
    switch_password = None
    switch_prompt = None
    threadLock = None
    _init_done = False

    ### user configs
    switch_vlan_name_prefix = ''

    def __init__(self, ip, login='admin', password='switch', prompt='->'):
        self.switch_ip = ip.strip()
        if len(self.switch_ip) == 0 :
            LOG.info("Init Error! Must provide a valid IP address!!!")
            return

        self.switch_login = login.strip()
        if len(self.switch_login) == 0 :
            self.switch_login = 'admin'
            
        self.switch_password = password.strip()
        if len(self.switch_password) == 0 :
            self.switch_password = 'switch'

        self.switch_prompt = prompt.strip()
        if len(self.switch_prompt) == 0 :
            self.switch_prompt = '->'

        self.aosapi = AOSAPI(AOSConnection(
                     self.switch_login,
                     self.switch_password,
                     self.switch_ip,
                     True,
                     True,
                     True,
                     -1,
                     None,
                     0,
                     False))
 
        self.threadLock = threading.Lock()
        self._init_done = True

    def set_config(self, vlan_name_prefix):
        self.switch_vlan_name_prefix = vlan_name_prefix

    def connect(self):
        if self._init_done == False :
            LOG.info("Driver is not initialized!!!")
            return False
        
        try:
            results = self.aosapi.login()
            if not self.aosapi.success():
                LOG.info("Login error %s: %s", self.switch_ip, results)
                return False
            else:
                return True
        except urllib2.HTTPError, e :
            self.aosapi.logout()
            LOG.info("Connect Error %s: %s", self.switch_ip, e)
            return False
        

    def disconnect(self):
        self.aosapi.logout()

    ###beware lock used!!!, dont call this func from another locked func
    def create_vpa(self, vlan_id, slotport, args=None):
        self.threadLock.acquire(1)
        ret = False
        if self.connect() == False:
            self.threadLock.release()
            return False

        ifindex = self._get_ifindex_from_slotport(slotport)
        results = self.aosapi.put('mib', 'vpaTable',
                        {'mibObject0':'vpaIfIndex:'+str(ifindex),
                         'mibObject1':'vpaVlanNumber:'+str(vlan_id),
                         'mibObject2':'vpaType:2'})['result']
        if self.aosapi.success():
            LOG.info("vpa %s --> %s created in %s successfully!", vlan_id, slotport, self.switch_ip)
            ret = True
        else:
            LOG.info("vpa %s --> %s creation in %s failed! %s", vlan_id, slotport, self.switch_ip, results)
        self.disconnect()
        self.threadLock.release()
        return ret

    ###beware lock used!!!, dont call this func from another locked func
    def delete_vpa(self, vlan_id, slotport, args=None): 
        self.threadLock.acquire(1)
        ret = False
        if self.connect() == False:
            self.threadLock.release()
            return False

        ifindex = self._get_ifindex_from_slotport(slotport)
        results = self.aosapi.delete('mib', 'vpaTable',
                        {'mibObject0':'vpaIfIndex:'+str(ifindex),
                         'mibObject1':'vpaVlanNumber:'+str(vlan_id)})['result']
        if self.aosapi.success():
            LOG.info("vpa %s --> %s deleted in %s successfully!", vlan_id, slotport, self.switch_ip)
            ret = True
        else:
            LOG.info("vpa %s --> %s deletion in %s failed! %s", vlan_id, slotport, self.switch_ip, results)
        self.disconnect()
        self.threadLock.release()
        return ret

    def create_vlan_locked(self, vlan_id, net_name=''):
        self.threadLock.acquire(1)
        ret = self.create_vlan(vlan_id, net_name)
        self.threadLock.release()
        return ret

    def create_vlan(self, vlan_id, net_name=''):
        ret = False
        if self.connect() == False:
            return False

        vlan_name = self.switch_vlan_name_prefix+'-'+net_name+'-'+str(vlan_id)
        results = self.aosapi.put('mib', 'vlanTable', 
                       {'mibObject0':'vlanNumber:'+str(vlan_id), 
                        'mibObject1':'vlanDescription:'+vlan_name})['result']
                        #'mibObject1':'vlanDescription:OpenStack-'+str(vlan_id)})['result']
        if self.aosapi.success():
            LOG.info("vlan %s created in %s successfully!", vlan_id, self.switch_ip)
            ret = True
        else:
            LOG.info("vlan %s creation in %s failed! %s", vlan_id, self.switch_ip, results)
        self.disconnect()
        return ret

    def delete_vlan_locked(self, vlan_id):
        self.threadLock.acquire(1)
        ret = self.delete_vlan(vlan_id)
        self.threadLock.release()
        return ret

    def delete_vlan(self, vlan_id):
        ret = False
        if self.connect() == False:
            return False

        results = self.aosapi.delete('mib', 'vlanTable', {'mibObject0':'vlanNumber:'+str(vlan_id)})['result']
        if self.aosapi.success():
            LOG.info("vlan %s deleted in %s successfully!", vlan_id, self.switch_ip)
            ret = True
        else:
            LOG.info("vlan %s deletion in %s failed! %s", vlan_id, self.switch_ip, results)
        self.disconnect()
        return ret

    def create_unp_vlan(self, vlan_id, args=None):
        ret = False
        if self.connect() == False:
            return False

        results = self.aosapi.put('mib', 'alaDaUserNetProfileTable',
                  {'mibObject0':'alaDaUserNetProfileName:' +'OpenStack-UNP-'+str(vlan_id),
                   'mibObject1':'alaDaUserNetProfileVlanID:'+str(vlan_id)})['result']
        if self.aosapi.success():
            LOG.info("unp_vlan %s creation in %s success!", vlan_id, self.switch_ip)
            ret = True
        else:
            LOG.info("unp_vlan %s creation in %s failed! %s", vlan_id, self.switch_ip, results)
        self.disconnect()
        return ret

    def create_unp_macrule(self, vlan_id, mac, args=None):
        ret = False
        if self.connect() == False:
            return False

        results = self.aosapi.put('mib', 'alaDaUNPCustDomainMacRuleTable',
                  {'mibObject0':'alaDaUNPCustDomainMacRuleAddr:'+str(mac),
                   'mibObject1':'alaDaUNPCustDomainMacRuleDomainId:0',
                   'mibObject2':'alaDaUNPCustDomainMacRuleProfileName:'
                                 +'OpenStack-UNP-'+str(vlan_id)}) ['result']
        if self.aosapi.success():
            LOG.info("unp_macrule[%s %s] creation in %s success!", vlan_id, mac, self.switch_ip)
            ret = True
        else:
            LOG.info("unp_macrule[%s %s] creation in %s failed! %s", vlan_id, mac, self.switch_ip, results)
        self.disconnect()
        return ret

    def get_unp_macrule(self, args=None):
        ret = None
        if self.connect() == False:
            return ret

        results = self.aosapi.query('mib', 'alaDaUNPCustDomainMacRuleTable',
                  {'mibObject0':'alaDaUNPCustDomainMacRuleAddr',
                   'mibObject1':'alaDaUNPCustDomainMacRuleDomainId',
                   'mibObject2':'alaDaUNPCustDomainMacRuleProfileName'})['result']

        if self.aosapi.success():
            ret = results
        else:
            LOG.info("get_unp_macrule failed in %s! [%s]", self.switch_ip, results)
        self.disconnect()
        return ret

    def create_unp_vlanrule(self, vlan_id):
        ret = False
        if self.connect() == False:
            return False

        results = self.aosapi.put('mib', 'alaDaUNPCustDomainVlanTagRuleTable',
                  {'mibObject0':'alaDaUNPCustDomainVlanTagRuleVlan:'+str(vlan_id),
                   'mibObject1':'alaDaUNPCustDomainVlanTagRuleDomainId:0',
                   'mibObject2':'alaDaUNPCustDomainVlanTagRuleVlanProfileName:'
                                 +'OpenStack-UNP-'+str(vlan_id)}) ['result']
        if self.aosapi.diag() == 200:
            LOG.info("unp_vlanrule[%s] creation in %s success!", vlan_id, self.switch_ip)
            ret = True
        else:
            LOG.info("unp_vlanrule[%s] creation in %s failed!", vlan_id, self.switch_ip, results)
        self.disconnect()
        return ret

    def delete_unp_vlan(self, vlan_id):
        ret = False
        if self.connect() == False:
            return False

        results = self.aosapi.delete('mib', 'alaDaUserNetProfileTable',
                  {'mibObject0':'alaDaUserNetProfileName:'+'OpenStack-UNP-'+str(vlan_id)})['result']
        if self.aosapi.success():
            LOG.info("unp_vlan %s deletion in %s success!", vlan_id, self.switch_ip)
            ret = True
        else:
            LOG.info("unp_vlan %s deletion in %s failed! %s", vlan_id, self.switch_ip, results)
        self.disconnect()
        return ret

    def delete_unp_macrule(self, vlan_id, mac):
        ret = False
        if self.connect() == False:
            return False

        results = self.aosapi.delete('mib', 'alaDaUNPCustDomainMacRuleTable',
                  {'mibObject0':'alaDaUNPCustDomainMacRuleAddr:'+str(mac),
                   'mibObject1':'alaDaUNPCustDomainMacRuleDomainId:0'})['result']
        if self.aosapi.success():
            LOG.info("unp_macrule[%s %s] deletion in %s suceess!", vlan_id, mac, self.switch_ip)
            ret = True
        else:
            LOG.info("unp_macrule[%s %s] deletion in %s failed! %s", vlan_id, mac, self.switch_ip, results)
            
        self.disconnect()
        return ret

    def delete_unp_vlanrule(self, vlan_id):
        ret = False
        if self.connect() == False:
            return False

        results = self.aosapi.delete('mib', 'alaDaUNPCustDomainVlanTagRuleTable',
                  {'mibObject0':'alaDaUNPCustDomainVlanTagRuleVlan:'+str(vlan_id),
                   'mibObject1':'alaDaUNPCustDomainVlanTagRuleDomainId:0'})['result']
        if self.aosapi.success():
            LOG.info("unp_vlanrule[%s] deletion in %s success!", vlan_id, self.switch_ip)
            ret = True
        else:
            LOG.info("unp_vlanrule[%s] deletion in %s failed!", vlan_id, self.switch_ip, results)
        self.disconnect()
        return ret

    def enable_stp_mode_flat(self):
        ret = False
        if self.connect() == False:
            return False

        results = self.aosapi.post('mib', 'vStpBridge',
                       {'mibObject0':'vStpBridgeMode:'+str(1)})['result']
        if self.aosapi.success():
            LOG.info("stp mode flat in %s success!", self.switch_ip)
            ret = True
        else:
            LOG.info("stp mode flat in %s failed! %s", self.switch_ip, results)
        self.disconnect()
        return ret

    def disable_stp_mode_flat(self):
        ret = False
        if self.connect() == False:
            return False

        results = self.aosapi.post('mib', 'vStpBridge',
                       {'mibObject0':'vStpBridgeMode:'+str(2)})['result']
        if self.aosapi.success():
            LOG.info("stp mode 1X1 in %s success!", self.switch_ip)
            ret = True
        else:
            LOG.info("stp mode 1X1 in %s failed! %s", self.switch_ip, results)
        self.disconnect()
        return ret

    def enable_mvrp_global(self):
        ret = False
        if self.connect() == False:
            return False

        results = self.aosapi.post('mib', 'alcatelIND1MVRPMIBObjects',
                  {'mibObject0':'alaMvrpGlobalStatus:'+str(1)}) ['result']
        if self.aosapi.success():
            LOG.info("mvrp enable global in %s success!", self.switch_ip)
            ret = True
        else:
            LOG.info("mvrp enable global in %s failed! %s", self.switch_ip, results)
        self.disconnect()
        return ret

    def disable_mvrp_global(self):
        ret = False
        if self.connect() == False:
            return False

        results = self.aosapi.post('mib', 'alcatelIND1MVRPMIBObjects',
                  {'mibObject0':'alaMvrpGlobalStatus:'+str(2)}) ['result']
        if self.aosapi.success():
            LOG.info("mvrp disable global in %s success!", self.switch_ip)
            ret = True
        else:
            LOG.info("mvrp disable global in %s failed! %s", self.switch_ip, results)
            #print results
        self.disconnect()
        return ret

    def enable_mvrp_if(self, slotport):
        ret = False
        if self.connect() == False:
            return False

        ifindex = self._get_ifindex_from_slotport(slotport)
        results = self.aosapi.post('mib', 'alaMvrpPortConfigTable',
                        {'mibObject0':'alaMvrpPortConfigIfIndex:'+str(ifindex),
                         'mibObject1':'alaMvrpPortStatus:'+str(1)})['result']
        if self.aosapi.success():
           LOG.info("mvrp enable on %s %s success!", slotport, self.switch_ip)
           ret = True
        else:
           LOG.info("mvrp enable on %s %s failed! %s", slotport, self.switch_ip, results)
        self.disconnect()
        return ret

    def disable_mvrp_if(self, slotport):
        ret = False
        if self.connect() == False:
            return False

        ifindex = self._get_ifindex_from_slotport(slotport)
        results = self.aosapi.post('mib', 'alaMvrpPortConfigTable',
                        {'mibObject0':'alaMvrpPortConfigIfIndex:'+str(ifindex),
                         'mibObject1':'alaMvrpPortStatus:'+str(2)})['result']
        if self.aosapi.success():
           LOG.info("mvrp disable on %s %s success!", slotport, self.switch_ip)
           ret = True
        else:
           LOG.info("mvrp disable on %s %s failed! %s", slotport, self.switch_ip, results)
           #print results
        self.disconnect()
        return ret

    def enable_mvrp(self, slotport=None):
        if slotport:
            return self.enable_mvrp_if(slotport)
        else:
            if self.enable_stp_mode_flat() == True:
                return self.enable_mvrp_global()
            else:
                return False

    def disable_mvrp(self, slotport=None):
        if slotport:
            return self.disable_mvrp_if(slotport)
        else:
            if self.disable_mvrp_global() == True:
                return self.disable_stp_mode_flat()
            else:
                return False


    def enable_unp(self, slotport):
        ret = False
        if self.connect() == False:
            return False
        ifindex = self._get_ifindex_from_slotport(slotport)
        results = self.aosapi.put('mib', 'alaDaUNPPortTable',
                        {'mibObject0':'alaDaUNPPortIfIndex:'+str(ifindex),
                         'mibObject1':'alaDaUNPPortClassificationFlag:'+str(1)})['result']
        if self.aosapi.success():
            LOG.info("unp enable on %s %s success!", slotport, self.switch_ip)
            ret = True
        else:
            LOG.info("unp enable on %s %s failed! [%s]", slotport, self.switch_ip, results)
        self.disconnect()
        return ret

    def disable_unp(self, slotport):
        ret = False
        if self.connect() == False:
            return False

        ifindex = self._get_ifindex_from_slotport(slotport)
        results = self.aosapi.delete('mib', 'alaDaUNPPortTable',
                        {'mibObject0':'alaDaUNPPortIfIndex:'+str(ifindex),
                         'mibObject1':'alaDaUNPPortClassificationFlag:'+str(1)})['result']
        if self.aosapi.success():
            LOG.info("unp disable on %s %s success!", slotport, self.switch_ip)
            ret = True
        else:
            LOG.info("unp disable on %s %s failed! [%s]", slotport, self.switch_ip, results)
            #print results
        self.disconnect()
        return ret

    def write_memory(self):
        ret = False
        if self.connect() == False:
            return False

        results = self.aosapi.post('mib', 'configManager',
                        {'mibObject0':'configWriteMemory:'+str(1)})['result']
        if self.aosapi.success():
            LOG.info("write memory success on %s", self.switch_ip)
            ret = True
        else:
            LOG.info("write memory failed on %s [%s]", self.switch_ip, results)

        self.disconnect()
        return ret


    def copy_running_certified(self):
        ret = False
        if self.connect() == False:
            return False

        results = self.aosapi.post('mib', 'chasControlModuleTable',
                        {'mibObject0':'entPhysicalIndex:'+str(65),
                         'mibObject1':'chasControlVersionMngt:'+str(2)})['result']
        if self.aosapi.success():
            LOG.info("copy running certified success on %s", self.switch_ip)
            ret = True
        else:
            results = self.aosapi.post('mib', 'chasControlModuleTable',
                        {'mibObject0':'entPhysicalIndex:'+str(66),
                         'mibObject1':'chasControlVersionMngt:'+str(2)})['result']
            if self.aosapi.success():
                LOG.info("copy running certified success on %s", self.switch_ip)
                ret = True
            else:
                LOG.info("copy running certified failed on %s [%s]", self.switch_ip, results)

        self.disconnect()
        return ret

   
    #####  OneTouch functions for OpenStack APIs #####
    def create_network(self, vlan_id, net_name=''):
        self.threadLock.acquire(1)
        ret = 0
        if self.create_vlan(vlan_id, net_name) == True :
            ret = self.create_unp_vlan(vlan_id)
        self.threadLock.release()
        return ret

    def delete_network(self, vlan_id):
        self.threadLock.acquire(1)
        ret = 0
        if self.delete_unp_vlan(vlan_id) == True :
            ret =  self.delete_vlan(vlan_id)
        self.threadLock.release()
        return ret

    def create_port(self, vlan_id, mac=None):
        self.threadLock.acquire(1)
        if mac :
            ret = self.create_unp_macrule(vlan_id, mac)
        else :
            ret = self.create_unp_vlanrule(vlan_id)
        self.threadLock.release()
        return ret

    def delete_port(self, vlan_id, mac=None):
        self.threadLock.acquire(1)
        if mac :
            ret = self.delete_unp_macrule(vlan_id, mac)
        else :
            ret = self.delete_unp_vlanrule(vlan_id)
        self.threadLock.release()
        return ret

    def save_config(self):
        self.threadLock.acquire(1)
        ret = 0
        if self.write_memory():
            time.sleep(1)
            ret = self.copy_running_certified()
            time.sleep(2)
            self.threadLock.release()
            return ret
        else:
            ret = False
        self.threadLock.release()
        return ret


    #####   Internal Utility functions #####
    
    def _get_ifindex_from_slotport(self, slotport):
        """ convert slot/port = '1/2' to ifIndex = 1002 """
        """ convert chassis/slot/port = '1/2/3' to ifIndex = 102003 """
        """ convert linkagg id = '5' to ifIndex = 40000005 """

        if len(slotport.split('/')) == 3 :
            chassis = int(slotport.split('/')[0])
            if chassis == 0:
                chassis = 1
            slot = int(slotport.split('/')[1])
            port = int(slotport.split('/')[2])
            return(str(((chassis-1)*100000) + (slot*1000) + port))
        elif len(slotport.split('/')) == 2 :
            slot = int(slotport.split('/')[0])
            port = int(slotport.split('/')[1])
            return(str((slot*1000) + port))
        elif len(slotport.split('/')) == 1 :
            linkagg = int(slotport.split('/')[0])
            return(str(40000000+linkagg))
        else:
            LOG.info("Error: ifIndex calc: invalid slotport %s",slotport)
            return 0


    ##### functions used by scripts outside of neutron-server

    # dont use lock in this API as it uses one-touch api which already has lock
    def clear_config(self, vlan_ids):
        # delete mac_rules
        results = self.get_unp_macrule()
        if len(results['data']):
            for i in vlan_ids:
                for key, value in  results['data']['rows'].items():
                    if 'OpenStack-UNP-'+str(i) == value['alaDaUNPCustDomainMacRuleProfileName'] :
                        self.delete_port(i, value['alaDaUNPCustDomainMacRuleAddr'])

        # delete vlan_rules and vlans
        for i in vlan_ids:
            self.delete_port(i)
            self.delete_unp_vlan(i)
            self.delete_vlan(i)

