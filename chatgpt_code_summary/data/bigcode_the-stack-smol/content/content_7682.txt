import json

from kazoo.client import KazooClient


zk = KazooClient(hosts='127.0.0.1:2181')
zk.start()
value = json.dumps({'host': '127.0.0.2', 'port': 8080}).encode()
zk.ensure_path('/demo')
r = zk.create('/demo/rpc', value, ephemeral=True, sequence=True)
print(r)
zk.stop()