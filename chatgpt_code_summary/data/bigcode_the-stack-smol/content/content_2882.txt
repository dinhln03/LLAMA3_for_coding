

from aws_cdk import (
    core,
    aws_iam as iam,
    aws_kinesis as kinesis,
    aws_kinesisfirehose as kinesisfirehose
)


class Lab07Stack(core.Stack):

    def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # The code that defines your stack goes here

        

        role01 = iam.CfnRole(self,id="firehose01_role",assume_role_policy_document= {
            "Statement": [{
            "Action": "sts:AssumeRole",
            "Effect": "Allow",
            "Principal": {
              "Service": "lambda.amazonaws.com"
            }
          }],
          "Version": "2012-10-17"
        },managed_policy_arns=[
            "arn:aws:iam::aws:policy/service-role/AWSLambdaKinesisExecutionRole"
            ])

        policy01=iam.CfnPolicy(self,id="firehose01_policy",policy_name="firehose01_policy",policy_document={
            'Version': "2012-10-17",
            'Statement': [
                {
                    "Action": [
                        's3:AbortMultipartUpload',
                        's3:GetBucketLocation',
                        's3:GetObject',
                        's3:ListBucket',
                        's3:ListBucketMultipartUploads',
                        's3:PutObject'
                        ],
                    "Resource": ['*'],
                    "Effect": "Allow"
                 }
            ]
        },roles=[role01.ref])

        

        


        delivery_stream = kinesisfirehose.CfnDeliveryStream(self, id = "firehose01",
                                            delivery_stream_name = "firehose01",
                                            extended_s3_destination_configuration = {
                                                # s3桶信息
                                                'bucketArn': 'arn:aws:s3:::fluent-bit-s3',
                                                
                                                # 压缩设置，老方案：gzip，新方案待定
                                                'compressionFormat': 'GZIP',
                                                # 格式转换，是否转换为orc，parquet，默认无
                                                'DataFormatConversionConfiguration':"Disabled",
                                                # 是否加密：默认无
                                                'EncryptionConfiguration':"NoEncryption",
                                                # 错误输出前缀
                                                'bufferingHints': {
                                                    'intervalInSeconds': 600,
                                                    'sizeInMBs': 128
                                                },
                                                'ProcessingConfiguration': {
                                                    "Enabled": True,
                                                    "Processor": {
                                                        "Type": "Lambda",
                                                        "Parameters": [
                                                            {
                                                                "ParameterName": "BufferIntervalInSeconds",
                                                                "ParameterValue": "60"
                                                            },
                                                            {
                                                                "ParameterName": "BufferSizeInMBs",
                                                                "ParameterValue": "3"
                                                            },
                                                            {
                                                                "ParameterName": "LambdaArn",
                                                                "ParameterValue": "arn:aws:lambda:ap-southeast-1:596030579944:function:firehose-test"
                                                             }
                                                        ]
                                                    }
                                                },
                                                'roleArn': 'arn:aws:iam::596030579944:role/avalon_lambda_kinesis_role',
                                                'S3BackupConfiguration': {
                                                    "BucketARN": 'arn:aws:s3:::fluent-bit-s3',
                                                    'bufferingHints': {
                                                        'intervalInSeconds': 600,
                                                        'sizeInMBs': 128
                                                    },
                                                    'compressionFormat': 'GZIP',
                                                    'EncryptionConfiguration':"NoEncryption",
                                                    'Prefix': "/backup",
                                                    'roleArn': 'arn:aws:iam::596030579944:role/avalon_lambda_kinesis_role'
                                                }
                                            },
                                            )
