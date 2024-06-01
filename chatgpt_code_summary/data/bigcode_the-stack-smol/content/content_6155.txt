# -*- coding: utf-8 -*-
"""
Created on Tue May 12 00:00:00 2020
@author: Shaji
"""

import boto3
import os
boto3.setup_default_session()

s3_client = boto3.client('s3')

def list_buckets(client=s3_client):
    """
    Usage: [arg1]:[initialized s3 client object],
    Description: Gets the list of buckets
    Returns: [list of buckets]
    """
    response = s3_client.list_buckets()
    buckets=[]
    for bucket in response['Buckets']:
        buckets.append(bucket["Name"])
    return buckets
    
def list_objects(bucket,prefix='',client=s3_client):
    """
    Usage: [arg1]:[bucket name],[arg2]:[pattern to match keys in s3],[arg3]:[initialized s3 client object],
    Description: Gets the keys in the S3 location
    Returns: [list of keys], [list of directories]
    """
    keys = []
    dirs = set()
    next_token = ''
    base_kwargs = {
        'Bucket':bucket,
        'Prefix':prefix,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            keys.append(k)
            dirs.add(k[:k.rfind('/')+1])
        next_token = results.get('NextContinuationToken')
    return keys,list(dirs)

def download_dir(bucket, prefix, local_path, client=s3_client):
    """
    Usage: [arg1]:[bucket name],[arg2]:[pattern to match keys in s3],[arg3]:[local path to folder in which to place files],[arg4]:[initialized s3 client object],
    Description: Downloads the contents to the local path
    """
    keys = []
    dirs = set()
    next_token = ''
    base_kwargs = {
        'Bucket':bucket,
        'Prefix':prefix,
    }
    local=local_path+bucket+'\\'
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            keys.append(k)
            dirs.add(k[:k.rfind('/')+1])
        next_token = results.get('NextContinuationToken')
    for d in dirs:
        dest_pathname = os.path.join(local, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
    for k in keys:
        dest_pathname = os.path.join(local, k)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        client.download_file(bucket, k, dest_pathname)
