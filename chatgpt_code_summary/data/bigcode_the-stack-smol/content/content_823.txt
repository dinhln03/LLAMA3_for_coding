#!/usr/bin/env python3
import boto3
s3_client = boto3.client('s3')
raw_response = s3_client.list_buckets()
for bucket in raw_response['Buckets']:
    print(bucket['Name'])
