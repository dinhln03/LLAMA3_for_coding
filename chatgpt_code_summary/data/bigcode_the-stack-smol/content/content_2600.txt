'''
Source code developed by DI2AG.
Thayer School of Engineering at Dartmouth College
Authors:    Dr. Eugene Santos, Jr
            Mr. Chase Yakaboski,
            Mr. Gregory Hyde,
            Dr. Keum Joo Kim
'''


import json
import argparse
import os
import sys
import pickle
import subprocess

from chp.query import Query

PASSED_JSON_FILE = '/home/cyakaboski/passed_message.json'
NODE = 'c-dell-m630-0-11'
SAVE_DIR = '/home/cyakaboski/temp'
BKB_PATHWAY_CORE_DIR = '/home/cyakaboski/src/python/projects/bkb-pathway-provider/core'
'''
PASSED_JSON_FILE = '/home/ncats/passed_message.json'
NODE = 'c-dell-m630-0-11'
SAVE_DIR = '/home/ncats/tmp'
BKB_PATHWAY_CORE_DIR = '/home/ncats/live/core'
'''
def processUiQuery(dict_):
    query_dict = dict()
    query_dict['name'] = dict_['name']
    query_dict['evidence'] = dict_['genetic_evidence']
    query_dict['targets'] = dict_['genetic_targets']
    if dict_['demographic_evidence'] is not None:
        query_dict['meta_evidence'] = [tuple(demo) for demo in dict_['demographic_evidence']]
    else:
        query_dict['meta_evidence'] = None
    if dict_['demographic_targets'] is not None:
        query_dict['meta_targets'] = [tuple(demo) for demo in dict_['demographic_targets']]
    else:
        query_dict['meta_targets'] = None

    query = Query(**query_dict)
    return query

def consumeJsonFile(file_name):
    with open(file_name, 'r') as passed_file:
        query_dict = json.load(passed_file)
    os.system('rm {}'.format(file_name))
    return query_dict

def runOnNode(query, node_name, save_dir):
    pickle_file, json_file = query.save(save_dir)
    command = ['ssh', node_name,
               'python3', os.path.join(BKB_PATHWAY_CORE_DIR, 'driver.py'),
               '--config_file', os.path.join(BKB_PATHWAY_CORE_DIR, 'driver.config'),
               '--headless',
               '--query_file', pickle_file,
               '--save_dir', save_dir]
    subprocess.run(command)

    return json_file

def makeVariableJsonFile(save_dir, node_name):
    vars_file = os.path.join(save_dir, 'bkb_variables.pk')
    command = ['ssh', node_name,
               'python3', os.path.join(BKB_PATHWAY_CORE_DIR, 'driver.py'),
               '--config_file', os.path.join(BKB_PATHWAY_CORE_DIR, 'driver.config'),
               '--get_variables', vars_file]
    subprocess.run(command)

    #--Collect vars_dict from vars_file
    with open(vars_file, 'rb') as f_:
        vars_dict = pickle.load(f_)
    return vars_dict

def collectResults(query_file):
    with open(query_file) as f_:
        query_res_dict = json.load(f_)
    return query_res_dict

def sendJson(results):
    print('Begin-JSON------')
    print(json.JSONEncoder().encode(results))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', default=None, type=str)
    parser.add_argument('--get_variables', action='store_true')

    args = parser.parse_args()

    if args.f is not None:
        #-- Consume JSON File passed by UI
        query_dict = consumeJsonFile(args.f)

        #-- Process the passed JSON file into recognized and runnable Query
        query = processUiQuery(query_dict)

        #-- Analyze the Query and run reasoning on a specified dell node.
        saved_query_file = runOnNode(query, NODE, SAVE_DIR)

        #-- Load JSON result file and send back over ssh
        res_json = collectResults(saved_query_file)
        sendJson(res_json)
    elif args.get_variables:
        vars_dict = makeVariableJsonFile(SAVE_DIR, NODE)
        sendJson(vars_dict)
