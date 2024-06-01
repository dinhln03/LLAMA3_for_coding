from databroker.v1 import from_config
from databroker.v0 import Broker
from .. import load_config

name = 'tes'
v0_catalog = Broker.from_config(load_config(f'{name}/{name}.yml'))
v1_catalog = from_config(load_config(f'{name}/{name}.yml'))
catalog = from_config(load_config(f'{name}/{name}.yml')).v2
