import json
import yaml
from jsonschema import validate
import os

configuration_file = os.environ['SC_ENABLER_CONF']

with open(configuration_file, 'r') as conf_file:
    input_config = yaml.safe_load(conf_file)

with open("./input_schema_validator.json", 'r') as schema_file:
    schema = json.load(schema_file)


def test_input_params():
    validate(instance=input_config, schema=schema)
