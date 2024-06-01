'''
Using aws fargate to run a fmriprep. Uses our own docker image, which contains a wrapper to download the data from S3 and push it back again.
Rhodri Cusack TCIN 2021-06, cusackrh@tcd.ie
'''

from ecs_control import register_task, run_task, wait_for_completion
import boto3
import msgpack
import msgpack_numpy as m

from os import path

def run_subjects(subjlist, input_bucket, do_wait=True):   
    response=[]
    for subj in subjlist:
        response.append(run_task(client, command = ['/usr/local/miniconda/bin/fmriprep-cusacklab.bash', input_bucket, subj, 'bids', 'deriv-2_topup']))
    
    if do_wait:
        wait_for_completion(client, response)
    
    return response
    

if __name__=='__main__':
    input_bucket='foundcog-adult-pilot'
    session = boto3.session.Session()
    client = session.client('ecs', region_name='eu-west-1')
    response = register_task(client) 
    print(response)
    subjlist = ['sub-06','sub-17','sub-03'] # subjects with small affine shifts between fMRI runs
    #subjlist =['sub-04','sub-02','sub-05','sub-07','sub-08','sub-09','sub-10','sub-11','sub-12','sub-13','sub-14','sub-15','sub-16']
    response = run_subjects(subjlist, input_bucket=input_bucket)
