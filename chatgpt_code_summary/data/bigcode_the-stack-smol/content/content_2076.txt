# -*- coding: utf-8 -*-
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.



from airflow import DAG
from airflow.contrib.operators.jenkins_job_trigger_operator import JenkinsJobTriggerOperator
from airflow.operators.python_operator import PythonOperator
from airflow.contrib.hooks.jenkins_hook import JenkinsHook

from six.moves.urllib.request import Request

import jenkins
from datetime import datetime
from datetime import timedelta


datetime_start_date = datetime(2018, 5, 3)
default_args = {
    "owner": "airflow",
    "start_date": datetime_start_date,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
    "concurrency": 8,
    "max_active_runs": 8
}


dag = DAG("test_jenkins", default_args=default_args, schedule_interval=None)
#This DAG shouldn't be executed and is only here to provide example of how to use the JenkinsJobTriggerOperator
#(it requires a jenkins server to be executed)

job_trigger = JenkinsJobTriggerOperator(
    dag=dag,
    task_id="trigger_job",
    job_name="red-beta-build-deploy",
    parameters={"BRANCH":"origin/master", "USER_ENV":"shameer"},
    #parameters="resources/paremeter.json", You can also pass a path to a json file containing your param
    jenkins_connection_id="jenkins_nqa" #The connection must be configured first
    )

def grabArtifactFromJenkins(**context):
    """
    Grab an artifact from the previous job
    The python-jenkins library doesn't expose a method for that
    But it's totally possible to build manually the request for that
    """
    hook = JenkinsHook("jenkins_nqa")
    jenkins_server = hook.get_jenkins_server()
    url = context['task_instance'].xcom_pull(task_ids='trigger_job')
    #The JenkinsJobTriggerOperator store the job url in the xcom variable corresponding to the task
    #You can then use it to access things or to get the job number
    #This url looks like : http://jenkins_url/job/job_name/job_number/
    url = url + "artifact/myartifact.xml" #Or any other artifact name
    self.log.info("url : %s", url)
    request = Request(url)
    response = jenkins_server.jenkins_open(request)
    self.log.info("response: %s", response)
    return response #We store the artifact content in a xcom variable for later use

artifact_grabber = PythonOperator(
    task_id='artifact_grabber',
    provide_context=True,
    python_callable=grabArtifactFromJenkins,
    dag=dag)

artifact_grabber.set_upstream(job_trigger)
