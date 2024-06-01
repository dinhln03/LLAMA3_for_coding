#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
# @Time    : 2020-01-16  15:53
# @Author  : 行颠
# @Email   : 0xe590b4@gmail.com
# @File    : view
# @Software: view
# @DATA    : 2020-01-16
"""


import os
import asyncio
import motor.motor_asyncio

import aiohttp
from aiohttp import web

import aiohttp_jinja2
import json
import jinja2
from apps.config import  *

import subprocess, shlex
from apps.ansible_config import get_ansible_hosts_data



@aiohttp_jinja2.template('package/package.html')
async def handler_package(request):
    return {'name': 'Andrew', 'age': 'Svetlov'}


@aiohttp_jinja2.template('package/workflow.html')
async def handler_package_workflow(request):
    return {'name': 'Andrew', 'age': 'Svetlov'}



async def handler_package_workflow_roles(request):
    post = await request.post()
    result = read_yaml_file(post['path'])

    roles = result[0]['roles']
    return web.json_response(roles)


async def handler_package_ansible_list(request):
    dir_path = request.app['settings'].ansible_package_workspace

    result = get_tree(dir_path)

    return web.json_response(result)


async def handler_package_ansible_read(request):
    post = await request.post()

    data = read_t_file(post['filename'])
    return web.json_response({"code": 1, "data": data})


async def handler_package_ansible_write(request):
    post = await request.post()

    data = write_t_file(post['path'], post['code'])
    return web.json_response({"code": 1, "data": data})


async def handler_package_ws_ansible_run(request):
    ws = web.WebSocketResponse()
    cmdb = request.app['cmdb']
    await ws.prepare(request)
    async for msg in ws:

        if msg.type == aiohttp.WSMsgType.TEXT:

            if msg.data == 'close':
                await ws.close()
            else:

                msg_data = json.loads(msg.data)

                params = []
                query_data = json.loads(msg_data['query'])
                for i in query_data['nodeDataArray']:

                    if i.get("role", None) and i.get("data", None):
                        params.append(i.get("data"))

                await ws.send_str(">> {} \r\n".format(json.dumps({"$or": params})))
                # await ws.send_json({"$or":params})

                import pprint
                pprint.pprint({"$or": params})
                result = cmdb.assets.find({"$or": params})

                hosts = []
                for host in await result.to_list(length=1000):
                    hosts.append(host)

                hosts_file = get_ansible_hosts_data(hosts)

                work_path = os.path.dirname(os.path.abspath(msg_data['path']))

                command = "ansible -i {} all -m ping".format(hosts_file)

                await ws.send_str(">> {} \r\n".format(command))

                await ws.send_str(">> {} \r\n".format("result:"))

                import time
                time.sleep(1)

                p = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, cwd=work_path)
                # 实时获取输出
                while p.poll() == None:
                    out = p.stdout.readline().strip()

                    err = p.stderr

                    if err:
                        # print("sub process err: ", err)
                        await ws.send_str(err.decode(encoding='utf-8', errors='strict'))

                    if out:
                        # print("sub process output: ", out)
                        await ws.send_str(out.decode(encoding='utf-8', errors='strict'))

                # 子进程返回值
                await ws.send_str("return code: {} ".format(p.returncode))

    return ws
