#!/usr/bin/env/python3

# Modified by contributors from Intel Labs

"""
Create json with config tag
"""
import os
import math

def create_json_file(config):
    config_file = os.path.join(os.environ['TVM_HOME'],'3rdparty/vta-hw/config/vta_config.json')
    vta_params = config.split('_')
    gemm_params = vta_params[0].split('x')
    batch = int(math.log(int(gemm_params[0]),2))
    blockIn = int(math.log(int(gemm_params[1]),2))
    blockOut = int(math.log(int(gemm_params[2]),2))
    try:
        fin = open(config_file, 'rt')
        data = fin.read()
        fin.close()
        data = data.replace('"LOG_BATCH" : 0', f'"LOG_BATCH" : {batch}')
        data = data.replace('"LOG_BLOCK_IN" : 4', f'"LOG_BLOCK_IN" : {blockIn}')
        data = data.replace('"LOG_BLOCK_OUT" : 4', f'"LOG_BLOCK_OUT" : {blockOut}')
        data = data.replace('"LOG_UOP_BUFF_SIZE" : 15', f'"LOG_UOP_BUFF_SIZE" : {vta_params[1]}')
        data = data.replace('"LOG_INP_BUFF_SIZE" : 15', f'"LOG_INP_BUFF_SIZE" : {vta_params[2]}')
        data = data.replace('"LOG_WGT_BUFF_SIZE" : 18', f'"LOG_WGT_BUFF_SIZE" : {vta_params[3]}')
        data = data.replace('"LOG_ACC_BUFF_SIZE" : 17', f'"LOG_ACC_BUFF_SIZE" : {vta_params[4]}')
    except IOError:
        print(f'Cannot open config file {config_file} for reading default config')
    new_config_file = os.path.join(os.environ['TVM_HOME'], '3rdparty/vta-hw/config', f'{config}.json')
    try:
      json_file = open(new_config_file, 'wt')
      json_file.write(data)
      json_file.close()
      print(f'New config written to {new_config_file}')
    except IOError:
      print(f'Cannot open config file {new_config_file} for writing new config')

if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser
    if len(sys.argv) < 2:
      sys.exit("At least 1 argument is required")
    else:
      if sys.argv[0].endswith('create_json.py'):
        ap = ArgumentParser(description='Create VTA json files with config and target')
        ap.add_argument('-c', '--config', type=str, default='1x16x16_15_15_18_17',
          help='VTA config (default: %(default)s)')
        args=ap.parse_args()
        create_json_file(args.config)
