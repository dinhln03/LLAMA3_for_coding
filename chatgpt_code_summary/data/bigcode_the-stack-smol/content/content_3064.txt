CLOUDFORMATION_TEMPLATE = """
AWSTemplateFormatVersion: '2010-09-09'

Parameters:
  s3FileName:
    Type: String

  environment:
    Type: String

  deploymentBucket:
    Type: String

Resources:
  # Place your AWS resources here
"""
