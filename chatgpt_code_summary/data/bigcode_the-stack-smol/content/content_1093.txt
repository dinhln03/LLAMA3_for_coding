##########################################################################
# If not stated otherwise in this file or this component's Licenses.txt
# file the following copyright and licenses apply:
#
# Copyright 2021 RDK Management
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################
'''
<?xml version="1.0" encoding="UTF-8"?><xml>
  <id/>
  <version>1</version>
  <name>TS_WANMANAGER_DSLWANoE_FixedMode_ActLink_P_1-1_WAN_Pri-Sec</name>
  <primitive_test_id/>
  <primitive_test_name>wanmanager_DoNothing</primitive_test_name>
  <primitive_test_version>1</primitive_test_version>
  <status>FREE</status>
  <synopsis>To check if  DSL line is active with FIXED_MODE  policy ,WAN Type and priorities being (1,1) (Primary,Secondary)  for DSL and WANOE respectively</synopsis>
  <groups_id/>
  <execution_time>40</execution_time>
  <long_duration>false</long_duration>
  <advanced_script>false</advanced_script>
  <remarks/>
  <skip>false</skip>
  <box_types>
    <box_type>Broadband</box_type>
  </box_types>
  <rdk_versions>
    <rdk_version>RDKB</rdk_version>
  </rdk_versions>
  <test_cases>
    <test_case_id>TC_WANMANAGER_55</test_case_id>
    <test_objective>This test case is to check if  DSL line is active with FIXED_MODE policy ,WAN Type and priorities being (1,1) (Primary,Secondary)  for DSL and WANOE respectively </test_objective>
    <test_type>Positive</test_type>
    <test_setup>Broadband</test_setup>
    <pre_requisite>1.Ccsp Components  should be in a running state else invoke cosa_start.sh manually that includes all the ccsp components and TDK Component
2.TDK Agent should be in running state or invoke it through StartTdk.sh script
3.WAN Manager should be enabled
4.Both DSL WAN and WANOE WAN connections should be available</pre_requisite>
    <api_or_interface_used>None</api_or_interface_used>
    <input_parameters>Device.X_RDK_WanManager.Policy 
Device.X_RDK_WanManager.CPEInterface.1.Wan.Type
Device.X_RDK_WanManager.CPEInterface.2.Wan.Type 
Device.X_RDK_WanManager.CPEInterface.1.Wan.Priority 
Device.X_RDK_WanManager.CPEInterface.2.Wan.Priority
Device.X_RDK_WanManager.CPEInterface.1.Wan.ActiveLink
Device.X_RDK_WanManager.CPEInterface.2.Wan.ActiveLink </input_parameters>
    <automation_approch>1.Load the Module
2.Get the current WAN Priority and WAN Types for DSL and WANOE interfaces
3.Make the priority and WAN Type unequal for further set operations to be success 
4.Get the current WAN policy , set the policy to FIXED_MODE if not in the  same policy
5.Set the  Wan Type and priorities as(1,1) (Primary, Secondary)  for DSL and WANOE respectively
6.Get the active link status for DSL and WANOE
7.With the current configurations  DSL Line is expected to be active
8.Revert the set values
9.Unload the module</automation_approch>
    <expected_output>With Fixed Mode policy Wan Type and priorities being (1,1) (Primary, Secondary)   for DSL and WANOE respectively - DSL Line is expected to be active </expected_output>
    <priority>High</priority>
    <test_stub_interface>WAN_MANAGER</test_stub_interface>
    <test_script>TS_WANMANAGER_DSLWANoE_FixedMode_ActLink_P_1-1_WAN_Pri-Sec</test_script>
    <skipped>No</skipped>
    <release_version>M90</release_version>
    <remarks>None</remarks>
  </test_cases>
</xml>

'''
# tdklib library,which provides a wrapper for tdk testcase script
import tdklib;
from tdkbVariables import *;
from time import sleep;
from WanManager_Utility import *;

obj = tdklib.TDKScriptingLibrary("tdkbtr181","RDKB");
obj1 = tdklib.TDKScriptingLibrary("sysutil","1");

#IP and Port of box, No need to change,
#This will be replaced with correspoing Box Ip and port while executing script
ip = <ipaddress>
port = <port>
obj.configureTestCase(ip,port,'TS_WANMANAGER_DSLWANoE_FixedMode_ActLink_P_1-1_WAN_Pri-Sec');
obj1.configureTestCase(ip,port,'TS_WANMANAGER_DSLWANoE_FixedMode_ActLink_P_1-1_WAN_Pri-Sec');

#Get the result of connection with test component and DUT
loadmodulestatus =obj.getLoadModuleResult();
loadmodulestatus1 =obj1.getLoadModuleResult();
print "[LIB LOAD STATUS]  :  %s" %loadmodulestatus;
print "[LIB LOAD STATUS]  :  %s" %loadmodulestatus1;

if "SUCCESS" in (loadmodulestatus.upper() and loadmodulestatus1.upper()):
    #Set the result status of execution
    obj.setLoadModuleStatus("SUCCESS");
    obj1.setLoadModuleStatus("SUCCESS");
    revertwantype =0;
    revertpriority =0;
    expectedresult="SUCCESS";

    tdkTestObj = obj.createTestStep('TDKB_TR181Stub_Get');
    defaultTypePriority,actualresult  = GetCurrentWanTypeAndPriority(tdkTestObj);
    if expectedresult in actualresult:
        #Set the result status of execution
        tdkTestObj.setResultStatus("SUCCESS");
        print "TEST STEP 1: Get the current WAN Type,Priority values for DSL and WANOE";
        print "EXPECTED RESULT 1: Should get the current WAN Type,Priority values for DSL and WANOE"
        print "ACTUAL RESULT 1 :The current WAN Type,Priority for DSL and WANOE are %s:"%defaultTypePriority;
        #Get the result of execution
        print "[TEST EXECUTION RESULT] : SUCCESS";

        step = 2;
        status, policy_initial = get_policy(tdkTestObj, step);
        if status == 0:
            tdkTestObj_Get = obj.createTestStep('TDKB_TR181Stub_Get');
            tdkTestObj_Set = obj.createTestStep('TDKB_TR181Stub_Set');
            print "***Checking if WAN types are equal and making them Unequal***";
            revertwantype,default,actualresult = MakeWANTypeUnEqual(tdkTestObj_Get,tdkTestObj_Set);
            if expectedresult in actualresult:
                tdkTestObj.setResultStatus("SUCCESS");
                print "***Checking if WAN priorities are equal and making them Unequal***";
                revertpriority,default,actualresult = MakePriorityUnEqual(tdkTestObj_Get,tdkTestObj_Set);
                if expectedresult in actualresult:
                    tdkTestObj.setResultStatus("SUCCESS");

                    #Set the Wan Manager Policy to FIXED_MODE
                    new_policy = "FIXED_MODE"
                    expectedresult="SUCCESS";
                    policyStatus =1;
                    revert = 0
                    if new_policy != policy_initial:
                       print "Setting the wanmanager policy to :%s"%new_policy
                       set_policy(new_policy, policy_initial, obj1, revert);
                       #Get the WANMANAGER POLICY and cross check with the Set value
                       step = step + 1;
                       status, policy = get_policy(tdkTestObj, step);
                       if status == 0:
                          revert = 1;
                          if policy == new_policy:
                             tdkTestObj.setResultStatus("SUCCESS");
                             print "The wanmanager policy is set successfully";
                             tdkTestObj = obj1.createTestStep('ExecuteCmd');
                             obj1.initiateReboot();
                             sleep(300);
                          else:
                              policyStatus =0;
                              tdkTestObj.setResultStatus("FAILURE");
                              print "The wanmanager policy is not set successfully";
                       else:
                           policyStatus =0;
                           tdkTestObj.setResultStatus("FAILURE");
                           print "Failed to get wanmanager policy after set ";

                    if policyStatus == 1:
                       print "The current WAN Manager Policy is %s" %new_policy;
                       wanDSL = "Primary";
                       wanWANOE = "Secondary";
                       priDSL = "1";
                       priWANOE ="1";
                       actualresult = SetWANTypethenPriority(tdkTestObj_Set,wanDSL,wanWANOE,priDSL,priWANOE);
                       revertwantype =1;
                       revertpriority =1;
                       if expectedresult in actualresult:
                           tdkTestObj.setResultStatus("SUCCESS");
                           print "TEST STEP 3: Set the (WANtype,Priority)for DSL(%s,%s) and WANOE(%s,%s)" %(wanDSL,priDSL,wanWANOE,priWANOE);
                           print "EXPECTED RESULT 3:Set operation is expected to be successful";
                           print "ACTUAL RESULT 3:set operations are successful";
                           #Get the result of execution
                           print "[TEST EXECUTION RESULT] : SUCCESS";

                           tdkTestObj = obj.createTestStep('TDKB_TR181Stub_Get');
                           tdkTestObj.addParameter("ParamName","Device.X_RDK_WanManager.CPEInterface.1.Wan.ActiveLink");
                           #Execute the test case in DUT
                           tdkTestObj.executeTestCase(expectedresult);
                           actualresult1 = tdkTestObj.getResult();
                           activeDSL = tdkTestObj.getResultDetails().strip().replace("\\n", "");

                           tdkTestObj = obj.createTestStep('TDKB_TR181Stub_Get');
                           tdkTestObj.addParameter("ParamName","Device.X_RDK_WanManager.CPEInterface.2.Wan.ActiveLink");
                           #Execute the test case in DUT
                           tdkTestObj.executeTestCase(expectedresult);
                           actualresult2 = tdkTestObj.getResult();
                           activeWANOE = tdkTestObj.getResultDetails().strip().replace("\\n", "");
                           if expectedresult in (actualresult1 and actualresult2):
                              #Set the result status of execution
                              tdkTestObj.setResultStatus("SUCCESS");
                              print "TEST STEP 4: Get the Active link status of DSL and WANOE";
                              print "EXPECTED RESULT 4: Active link status of DSL and WANOE should be fetched successfully";
                              print "ACTUAL RESULT 4: Get operation succeeded";
                              #Get the result of execution
                              print "[TEST EXECUTION RESULT] : SUCCESS";

                              if activeDSL == "true" and activeWANOE == "false":
                                  #Set the result status of execution
                                  tdkTestObj.setResultStatus("SUCCESS");
                                  print "TEST STEP 5: Get the Active link status of DSL and WANOE";
                                  print "EXPECTED RESULT 5: Active link status of DSL is expected to be true and WANOE as false";
                                  print "ACTUAL RESULT 5: DSL status :%s, WANOE status : %s" %(activeDSL,activeWANOE);
                                  #Get the result of execution
                                  print "[TEST EXECUTION RESULT] : SUCCESS";
                              else:
                                  #Set the result status of execution
                                  tdkTestObj.setResultStatus("FAILURE");
                                  print "TEST STEP 5: Get the Active link status of DSL and WANOE";
                                  print "EXPECTED RESULT 5:Active link status of DSL is expected to be true and WANOE as false";
                                  print "ACTUAL RESULT 5: DSL status :%s, WANOE status : %s" %(activeDSL,activeWANOE);
                                  #Get the result of execution
                                  print "[TEST EXECUTION RESULT] : FAILURE";
                           else:
                               #Set the result status of execution
                               tdkTestObj.setResultStatus("FAILURE");
                               print "TEST STEP 4: Get the Active link status of DSL and WANOE";
                               print "EXPECTED RESULT 4: Active link status of DSL is expected to be true and WANOE as false";
                               print "ACTUAL RESULT 4: Get operation failed ";
                               #Get the result of execution
                               print "[TEST EXECUTION RESULT] : FAILURE";
                           if revert == 1:
                              set_policy(new_policy, policy_initial, obj1, revert);
                       else:
                           tdkTestObj.setResultStatus("FAILURE");
                           print "TEST STEP 3: Set the (WANtype,Priority)for DSL(%s,%s) and WANOE(%s,%s)"%(wanDSL,priDSL,wanWANOE,priWANOE);
                           print "EXPECTED RESULT 3:Set operation is expected to be successful";
                           print "ACTUAL RESULT 3 :set operations failed";
                           #Get the result of execution
                           print "[TEST EXECUTION RESULT] : FAILURE";
                    else:
                         print "set operation of WAN Policy failed";
                else:
                    tdkTestObj.setResultStatus("FAILURE");
                    print "Unable to make WAN priorities Un-equal"
            else:
                tdkTestObj.setResultStatus("FAILURE");
                print "Unable to make WAN Types Un-equal"
        else:
            tdkTestObj.setResultStatus("FAILURE");
            print "The current policy is not the expected policy";
    else:
        #Set the result status of execution
        tdkTestObj.setResultStatus("FAILURE");
        print "TEST STEP 1: Get the default WAN Type,Priority values for DSL and WANOE";
        print "EXPECTED RESULT 1: Should get the default WAN Type,Priority values for DSL and WANOE"
        print "ACTUAL RESULT 1 :The default WAN Type,Priority for DSL and WANOE are %s:"%defaultTypePriority;
        #Get the result of execution
        print "[TEST EXECUTION RESULT] : FAILURE";

    #Revert operations
    revertflag =1;
    if revertpriority ==1:
        print "Reverting priority to defaults";
        paramList = ["Device.X_RDK_WanManager.CPEInterface.1.Wan.Priority","Device.X_RDK_WanManager.CPEInterface.2.Wan.Priority"];
        tdkTestObj = obj.createTestStep('TDKB_TR181Stub_Set');
        index = 2;
        for item in paramList:
            tdkTestObj.addParameter("ParamName",item);
            tdkTestObj.addParameter("ParamValue",defaultTypePriority[index]);
            tdkTestObj.addParameter("Type","int");
            expectedresult= "SUCCESS";
            #Execute testcase on DUT
            tdkTestObj.executeTestCase(expectedresult);
            result = tdkTestObj.getResult();
            Setresult = tdkTestObj.getResultDetails();
            index =index +1;
            if expectedresult in result:
               tdkTestObj.setResultStatus("SUCCESS");
            else:
                revertflag =0;
                print "Revert operation failed for WAN priority";
                tdkTestObj.setResultStatus("FAILURE");
                break;

    if revertwantype == 1:
        print "Reverting WAN Type to defaults";
        paramList = ["Device.X_RDK_WanManager.CPEInterface.1.Wan.Type","Device.X_RDK_WanManager.CPEInterface.2.Wan.Type"];
        tdkTestObj = obj.createTestStep('TDKB_TR181Stub_Set');
        index = 0;
        for item in paramList:
            tdkTestObj.addParameter("ParamName",item);
            tdkTestObj.addParameter("ParamValue",defaultTypePriority[index]);
            tdkTestObj.addParameter("Type","string");
            expectedresult= "SUCCESS";
            #Execute testcase on DUT
            tdkTestObj.executeTestCase(expectedresult);
            result = tdkTestObj.getResult();
            Setresult = tdkTestObj.getResultDetails();
            index =index +1;
            if expectedresult in result:
               tdkTestObj.setResultStatus("SUCCESS");
            else:
                revertflag =0;
                print "Revert operation failed for WAN Type";
                tdkTestObj.setResultStatus("FAILURE");
                break;

    #printing the final revert status
    if revertflag == 1:
       print "Revert operation successful for WAN Type and WAN priority";
    else:
        print "Revert operation failed for either WAN Type or WAN priority";

    obj.unloadModule("tdkbtr181");
    obj1.unloadModule("sysutil");
else:
    print "Failed to load module";
    obj.setLoadModuleStatus("FAILURE");
    obj1.setLoadModuleStatus("FAILURE");
