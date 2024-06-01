#!/usr/bin/env python
# -*- coding:utf-8 _*-
# @author  : Lin Luo / Bruce Liu
# @time    : 2020/1/3 21:35
# @contact : 15869300264@163.com / bruce.w.y.liu@gmail.com
import argparse

parser = argparse.ArgumentParser()
parser.add_argument_group()
parser.add_argument('-c', '--config', help='config file for run and operation', required=False)
group = parser.add_mutually_exclusive_group()
group.add_argument('-a', '--add', help='add sk with ip', required=False)
group.add_argument('-d', '--delete', help='delete sk by sk or ip', required=False)
# group.add_argument('-e', '-examine', help='examine the status of ip', required=False)
group.add_argument('-r', '--run', help='run the main project', action='store_true', required=False)
group.add_argument('-t', '--test', help='test the config file, default path is conf/odyn.conf', action='store_true',
                   required=False)
group.add_argument('-s', '--stop', help='stop the main project', action='store_true', required=False)
args = parser.parse_args()
