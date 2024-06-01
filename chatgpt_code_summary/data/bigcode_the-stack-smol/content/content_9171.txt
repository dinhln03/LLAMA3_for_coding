import boto3


MAX_OBJECT_LIMIT = 1000


def get_s3_client():
    boto3.setup_default_session()
    return boto3.client('s3')


def is_object_name(client, bucket, name):
    try:
        client.get_object(Bucket=bucket, Key=name)
        return True
    except:
        return False


def is_bucket_name(client, name):
    response = client.list_buckets()['Buckets']
    for r in response:
        if name == r['Name']:
            return True
    return False


def get_list_of_bucket_names(client):
    response = client.list_buckets()['Buckets']
    return map(lambda x: x['Name'], response)


def get_list_of_objects(client, bucket):
    list_of_objects = []
    response = client.list_objects_v2(Bucket=bucket, MaxKeys=MAX_OBJECT_LIMIT)
    list_of_objects.extend(get_keys_from_response(response))
    token = get_token_from_response(response)

    while len(token) > 0:
        response = client.list_objects_v2(Bucket=bucket, MaxKeys=MAX_OBJECT_LIMIT, ContinuationToken=token)
        list_of_objects.extend(get_keys_from_response(response))
        token = get_token_from_response(response)

    return list_of_objects


def get_token_from_response(response):
    try:
        return response['NextContinuationToken']
    except KeyError:
        return ''


def get_keys_from_response(response):
    try:
        return map(lambda x: x['Key'], response['Contents'])
    except KeyError:
        return []


def get_s3_object_as_string(client, bucket, key):
    s3_object = client.get_object(Bucket=bucket, Key=key)
    return s3_object['Body'].read()


def put_local_to_s3(client, local, bucket, key):
    with open(local, 'rb') as local_file:
        file_as_bytes = local_file.read()
        put_s3_object(client, bucket, key, file_as_bytes)


def put_s3_object(client, bucket, key, body):
    client.put_object(Body=body, Bucket=bucket, Key=key)


def put_empty_s3_object(client, bucket, key):
    client.put_object(Body=b'', Bucket=bucket, Key=key)


def copy_object_between_buckets(client, source, bucket, key):
    client.copy_object(Bucket=bucket, Key=key, CopySource=source)


def download_s3_file(bucket, key, location):
    try:
        s3 = boto3.resource('s3')
        s3.Bucket(bucket).download_file(key, location)
    except:
        exit(1)


def create_bucket(client, name):
    print('got ' + name)
    response = client.create_bucket(Bucket=name)
    if response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print('Could not create bucket')
        exit(1)


def delete_s3_bucket(client, name):
    response = client.delete_bucket(Bucket=name)
    if response['ResponseMetadata']['HTTPStatusCode'] != 204:
        print('Could not delete bucket')
        exit(1)


def delete_s3_object(client, bucket, key):
    client.delete_object(Bucket=bucket, Key=key)
