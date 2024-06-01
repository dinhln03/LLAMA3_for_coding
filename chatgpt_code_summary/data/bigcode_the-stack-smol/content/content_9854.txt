# !/usr/bin/env python
# -*-coding: utf-8 -*-

__author__ = 'wtq'

LOG_PATH = "monitor_logging.log"

REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379

# 采集的间隔与间断时长
MONITOR_INTERVAL = 1
MONITOR_PEROID = 3

# 监控的读写速率的网卡
NET_NAME = 'eth0'

# 系统内各台机器的名字，以此来计算系统的平均负载信息
SYSTEM_MACHINE_NAME = ["storage1", "storage2"]

# 用来计算客户端链接数的机器名字，一般为master
CLIENT_LINK_MACNHIE = ["storage1"]

DISK_ALL_SPACE = 100
CPU_KERNEL_NUMS = 32
MEM_ALL_SPACE = 100

FASTDFSPORT = '8000'
REDIS_SYSTEM_KEY = 'system'

FASTDFS_PEROID = 3
