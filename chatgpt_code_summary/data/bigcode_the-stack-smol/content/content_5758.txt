import argparse
import boto3
import json
from uuid import uuid4
import os

S3_BUCKET = os.environ["S3_BUCKET"]
S3_BUCKET_KEY_ID = os.environ["S3_BUCKET_KEY_ID"]
S3_BUCKET_KEY = os.environ["S3_BUCKET_KEY"]
AZ_PROCESSED_FILE = "/mnt/aws-things-azure-processed.json"

if __name__ == '__main__':

    client = boto3.client(
        'iot',
        region_name=os.environ["AWS_REGION"],
        aws_access_key_id=os.environ["AWS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_KEY"]
    )

    with open (AZ_PROCESSED_FILE) as file:
        jsonJobDoc = json.load(file)
    
    for thing in jsonJobDoc['things']:
        print (thing['thingName'])
        print (thing['thingArn'])
        print (thing['azure']['iotconnstr'])

        response = client.create_job(
            jobId='upgrade-'+thing['thingName'] + "-" + str(uuid4()),
            targets=[
                thing['thingArn'],
            ],
            document="{ \"operation\": \"upgradetoAzure\", \"fileBucket\": \""+S3_BUCKET+"\", \"ACCESS_KEY\": \""+S3_BUCKET_KEY_ID+ "\",\"SECRET_KEY\": \""+S3_BUCKET_KEY+ "\", \"AZURE_CONNECTION_STRING\": \""+thing['azure']['iotconnstr'] + "\" }",
            jobExecutionsRolloutConfig={
                'maximumPerMinute': 5,
                'exponentialRate': {
                    'baseRatePerMinute': 5,
                    'incrementFactor': 1.1,
                    'rateIncreaseCriteria': {
                        'numberOfNotifiedThings': 1
                    }
                }
            },
            abortConfig={
                'criteriaList': [
                    {
                        'failureType': 'FAILED',
                        'action': 'CANCEL',
                        'thresholdPercentage': 100,
                        'minNumberOfExecutedThings': 1
                    },
                ]
            }    
        )