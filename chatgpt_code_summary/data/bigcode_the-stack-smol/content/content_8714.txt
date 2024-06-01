import boto
from boto.dynamodb2.fields import HashKey
from boto.dynamodb2.table import Table
conn = boto.dynamodb.connect_to_region('us-west-2')
connection=boto.dynamodb2.connect_to_region('us-west-2')
users = Table.create('users', schema=[
     HashKey('username'), # defaults to STRING data_type
 ], throughput={
     'read': 5,
     'write': 15,
 }, 
)
consumerTable.put_item({"username":"user"})
