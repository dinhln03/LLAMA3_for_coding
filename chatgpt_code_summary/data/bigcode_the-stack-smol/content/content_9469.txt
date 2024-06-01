# Copyright (c) 2022 RWTH Aachen - Werkzeugmaschinenlabor (WZL)
# Contact: Simon Cramer, s.cramer@wzl-mq.rwth-aachen.de

from sherpa import Client
from sherpa.schedulers import Scheduler, _JobStatus
import requests
import json
import logging as logg
import numpy as np
import socket
from time import sleep
import os
from absl import logging
from s3_smart_open import read_json, to_s3, get_filenames, generate_s3_strings, generate_s3_session, delete_s3_objects 

logger = logg.getLogger(__name__)

class ArgoScheduler(Scheduler):
    """Argo Scheduler submit, update, kill jobs and send metrics for sherpa hpo

    Args:
        Scheduler (class): shepra.schedulers
    """    
    def __init__(self, default_parameter,trial_run_parameter,lower_is_better,objective,filename_objective,argo_ip,argo_port,k8_namespace,storage_strategy='keep',output_dir=''):
        """Set init values

        Args:
            default_parameter (dict): Parameter that will be submitted with the argo workflow in a kind of input flags.
            rebuild_parameter (dict): Parameter that were genereted when creating the hp for sherpa.
            lower_is_better (bool): whether to minimize or maximize the objective
            objective (str): Name of the objective that will be optimized for. Must be a key/name from the metrics that were generated within a trial run.
            filename_objective (str): Filename of the file that contains the objective value which was created within a trial run.
            argo_ip (str): Argo server ip
            argp_port (str): Argo server port
            k8_namespace (str): Name of the kubernetes namespace where the trial container should be executed.
            storage_strategy (str, optional): wether to keep all, delete all or keep the files from the best run. Defaults to 'keep'.
            output_dir (str): needed for sherpa api
        """        
        # Load Argo Api Token from env variable (set by k8 secret)
        if 'api_exec_token' in os.environ:
            api_token =  'Bearer ' + os.environ['api_exec_token']
        else:
            logging.error('No Authorization Token detected. Check Kubernetes Secrets and Argo Template!')

        logging.info('Default Parameter: {}'.format(default_parameter))

        self.submit_url = 'https://' + argo_ip + ':' +  argo_port + '/api/v1/workflows/' + k8_namespace + '/submit'
        self.status_url = 'https://' + argo_ip + ':' +  argo_port + '/api/v1/workflows/' + k8_namespace + '/'
        self.delete_url = self.status_url
        self.client = Client()
        self.best_metric = {"job_id": None, "metric": None}
        self.headers = {'Authorization': api_token }
        self.killed_jobs = []
        self.output_dir = output_dir
        self.default_parameter = default_parameter
        self.trial_run_parameter = trial_run_parameter
        self.storage_strategy = storage_strategy
        self.hostname = socket.gethostname()
        self.trials = {}
        self.run_name = self.default_parameter['run_name']
        self.metrics_filename = filename_objective
        self.objective =  objective
        self.lower_is_better = lower_is_better
        self.output_path = self.default_parameter['output_path']

        self.decode_status = {'Succeeded': _JobStatus.finished,
                              'Running': _JobStatus.running,
                              'Pending': _JobStatus.queued,
                              'Failed': _JobStatus.failed,
                              'Stopped': _JobStatus.killed,
                              'Other': _JobStatus.other}

    def make_request(self,request,data,max_wait=600,step=5,wait=0):
        """Sends a get, post or delete request every step seconds until the request was successful or wait exceeds max_wait.

        Args:
            request (str): Define which kind of request to execute.
            data (str): Submit information or sherpas job_id for a status request or job_id for deleting a trial.
            max_wait (int, optional): Time in seconds after which the requests repetition will be stopped. Defaults to 600.
            step (int, optional): Time in seconds after which a faulty request is repeated. Defaults to 5.
            wait (int, optional): Variable to which the step time is added and compared to max_wait. Defaults to 0.

        Returns:
            [class]: Response
        """    

        proxies = {"http": None, "https": None}

        if request == 'GET':
            response = requests.get(self.status_url+data, headers=self.headers, proxies=proxies, verify=False)
        elif request == 'POST':
            response = requests.post(self.submit_url, headers=self.headers, data=data, proxies=proxies, verify=False)
        elif request == 'DELETE':
            response =  requests.delete(self.status_url+data, headers=self.headers, proxies=proxies, verify=False)
        else:
            logging.error('Request argument is none of ["GET","POST","DELETE"].')

        if response.status_code == 200 or wait > max_wait:
            if wait > max_wait:
                logging.warning("Request has failed for {} seconds with status code: {}:{}".format(max_wait, response.status_code, response.reason))
            return response
        else:
            sleep(step)
            logging.error("Request has failed for {} times with reason {}:{}".format(1+int((max_wait/step)-((max_wait/step)-(wait/step))), response.status_code, response.reason))
            return self.make_request(request=request,data=data,max_wait=max_wait,step=step,wait=wait+step)


    def file_strategy(self,job_id,metrics):
        """Delete all trial files which were generated through a hpo trial
            It deletes all files in the output_path related to the job_id

        Args:
            job_id (str): Sherpa Job_ID / Argo trial workflow name
            metrics (dict): metrics to compare
        """ 
        if job_id in self.trials:
            trial = self.trials[job_id]
            if 'output_path' in trial:
                if self.storage_strategy == 'delete':
                    delete_s3_objects(trial['output_path'])
                elif self.storage_strategy == 'best':
                    if self.best_metric['metric'] == None:
                        self.best_metric['metric'] = metrics[self.objective]
                        self.best_metric['job_id'] = job_id
                    elif self.lower_is_better == True and metrics[self.objective] < self.best_metric['metric']:
                        delete_s3_objects(self.trials[self.best_metric['job_id']]['output_path'])
                        self.best_metric['metric'] = metrics[self.objective]
                        self.best_metric['job_id'] = job_id
                        logging.info('New best trial {} with metric {}'.format(self.best_metric['job_id'],
                                                                               self.best_metric['metric']))
                    elif self.lower_is_better == False and metrics[self.objective] > self.best_metric['metric']:
                        delete_s3_objects(self.trials[self.best_metric['job_id']]['output_path'])
                        self.best_metric['metric'] = metrics[self.objective]
                        self.best_metric['job_id'] = job_id
                        logging.info('New best trial {} with metric {}'.format(self.best_metric['job_id'],
                                                                               self.best_metric['metric']))
                    else:
                        delete_s3_objects(trial['output_path'])  

             
    def submit_job(self,command, env={}, job_name=''):
        """Submits a new hpo trial to argo in order to start a workflow template

        Args:
            command (list[str]): List that contains ['Argo WorkflowTemplate','Entrypoint of that Argo WorkflowTemplate]
            env (dict, optional): Dictionary that contains env variables, mainly the sherpa_trial_id. Defaults to {}.
            job_name (str, optional): Not needed for Argo scheduler. Defaults to ''.

        Returns:
            [str]: Sherpa Job_ID / Name of the workflow that was started by Argo
        """        
        os.environ['SHERPA_TRIAL_ID'] = env['SHERPA_TRIAL_ID']
        # Get new trial from the DB
        trial = self.client.get_trial()
        tp = trial.parameters
        WorkflowTemplate = command[0]
        entrypoint = command[1]
        # Set output path for next trial by using sherpa trial ID, Trial ID --> 0,1,...max_num_trials-1 or trial parameters 'save_to', 'load_from' for PBT and ASHA. 
        default_parameter = self.default_parameter

        if 'save_to' in tp:
            default_parameter['output_path'] = os.path.join(self.output_path,str(tp['save_to']),'')
        else:
            default_parameter['output_path'] = os.path.join(self.output_path,str(env['SHERPA_TRIAL_ID']),'')

        if 'load_from' in tp and tp['load_from'] != '':
            default_parameter['model_input_path'] = os.path.join(self.output_path,str(tp['load_from']),'')
            WorkflowTemplate = eval(self.trial_run_parameter)['WorkflowTemplateContinue']
            entrypoint = eval(self.trial_run_parameter)['EntrypointContinue']
        else:
            default_parameter['model_input_path'] = ''

        # Merge trial parameter with the default parameter dict
        # Eval trial_run_parameter string
        merged_parameter = eval(self.trial_run_parameter)
        for k, v in default_parameter.items():
            merged_parameter[k] = v

        # Get epochs to save them in trials in order to load epochs when sending metrics to the DB (self.get_status()) 
        epochs = merged_parameter.get('epochs',0)

        # Convert the parameter dictionary to a list format for the input parameter for argo
        parameters_list = []
        for key, val in merged_parameter.items():
            parameters_list.append("{}={}".format(key,val))

        # The data that will be submitted to argo. The Label will make it easier to filter the workflows matching to the sherpa Workflow
        data = json.dumps( {"resourceKind": "WorkflowTemplate", 
                            "resourceName": WorkflowTemplate, 
                            "submitOptions": {"entrypoint": entrypoint,
                                              "labels" : "sherpa_run="+self.hostname+",run_name="+self.run_name,
                                              "parameters" : parameters_list }})

        # Submits the WorkflowTemplate with the data to Argo
        response_submit = self.make_request(request='POST',data=data)
        
        # A successfully submitted workflow will have a response_status_code == 200
        if response_submit.status_code == 200:
            job_id =  json.loads(response_submit.content)['metadata']['name']
            logging.info('Submitted trial {} with job_id {}'.format(env['SHERPA_TRIAL_ID'],job_id))
        else:
            job_id = 'failed_trial_id_' + str(env['SHERPA_TRIAL_ID'])
            logging.warning('Failed to sumbit job with Trial_ID {} to argo.'.format(env['SHERPA_TRIAL_ID']))
        
        # Save some trial information which is needed in self.get_status
        self.trials[job_id] = {'trial': trial,'epochs':epochs,'output_path':default_parameter['output_path'],'model_input_path':default_parameter['model_input_path'],'status':0,'finished':False}

        # return the Argo workflow name to sherpa
        return job_id

    def get_status(self, job_id):
        """Obtains the current status of the job.
            Sends objective values/metrics to the DB when a trial succeeded.
            Compares objective values and decides wether to delete or keep files. 

        Args:
            job_id (str): Sherpa Job_ID / Name of the workflow that was started by Argo

        Returns:
            sherpa.schedulers._JobStatus: the job-status.
        """        

        response_status = self.make_request(request='GET',data=job_id)
        if response_status.status_code == 200:
            status = json.loads(response_status.content)['status']['phase']
            # sends metric to DB when dag has finished
            if status == 'Succeeded':
                # When Argo trial dag has succeeded load metrics and keep/delete files in s3 storage. Set finished flag to true afterwards in order to return Succeeded Status to shera in the next runner_loop()
                if self.trials[job_id]['finished'] == True: 
                    logging.info('Set status to finished for trial : {}'.format(self.trials[job_id]['trial'].id))
                else:
                    filename = self.metrics_filename
                    input_path = self.trials[job_id]['output_path']
                    metrics = read_json(input_path,filename)
                    logging.info('Send metrics for trial: {}'.format(self.trials[job_id]['trial'].id))
                    self.client.send_metrics(trial=self.trials[job_id]['trial'], iteration=self.trials[job_id]['epochs'],objective=metrics[self.objective],context=metrics)
                    status = 'Running'
                    # Set finished Flag to true after client send metrics to the DB
                    self.trials[job_id]['finished'] = True
                    # Delete all files / keep all files / keep the files of the best run in s3 storage
                    self.file_strategy(job_id,metrics)
            elif status == 'Failed':
                delete_s3_objects(self.trials[job_id]['output_path'])

        elif job_id in self.killed_jobs:
            status = 'Stopped'
        else:
            status = 'Other'

        # Decode Job status
        s = self.decode_status.get(status, _JobStatus.other)
        # info when job status has changed
        if s != self.trials[job_id]['status']:
            logging.info('Jobstatus: {} for Job {}'.format(status,job_id))
        self.trials[job_id]['status'] = s

        return s

        
    def kill_job(self, job_id):
        """Kill a job by deleting the argo workflow completly

        Args:
            job_id (str): Sherpa Job_ID / Name of the workflow that was started by Argo
        """        
        # Delete Argo Trial Dag / Workflow
        response_kill = self.make_request(request='DELETE',data=job_id)
        if response_kill.status_code == 200:
            # Append killed workflows job_id to list in order to return Status killed in get status because the request will not succeed (workflow does not exsist anymore)
            self.killed_jobs.append(str(job_id))
