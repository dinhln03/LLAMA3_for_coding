# coding=utf-8
import sys
import traceback
import numpy as np
import os
import Putil.base.logger as plog

plog.PutilLogConfig.config_file_handler(filename='./test/data/_log_test_common_data_multiprocess.log', mode='w')
plog.PutilLogConfig.config_log_level(stream=plog.INFO, file=plog.DEBUG)
plog.PutilLogConfig.config_format(plog.FormatRecommend)
plog.PutilLogConfig.config_handler(plog.stream_method | plog.file_method)
logger = plog.PutilLogConfig('TesCommonData').logger()
logger.setLevel(plog.DEBUG)
MainLogger = logger.getChild('Main')
MainLogger.setLevel(plog.DEBUG)

import Putil.test.data.test_common_data_unit as tbase
import Putil.data.common_data as pcd
import multiprocessing

pcd.DataPutProcess.set_running_mode(pcd.DataPutProcess.RunningMode.Debug)

if __name__ == '__main__':
    manager_common_data = pcd.CommonDataManager()
    manager_common_data.start()
    data = manager_common_data.TestCommonData()

    manager = multiprocessing.Manager()
    pool = multiprocessing.Pool()

    dpq = pcd.DataPutProcess(data, manager, pool)
    pool.close()

    dq = dpq.DataQueue()

    restart_param = dict()

    restart_param['critical_process'] = 'random_fill'
    dpq.restart(**restart_param)

    # pool.join()
    # print(dpq.queue_process_ret.get())

    count = 0
    while dpq.has_next():
        data = dq.get()
        assert len(data) == 1
        for k, v in enumerate(data[0]):
            assert v.datas().shape[0] == 1
            pass
        count += 1
        pass
    assert count == 100

    restart_param['device_batch'] = [1]
    restart_param['critical_process'] = 'random_fill'
    dpq.restart(**restart_param)
    count = 0
    while dpq.has_next():
        dq.get()
        count += 1
        pass
    assert count == 100

    restart_param['device_batch'] = [1]
    restart_param['critical_process'] = 'allow_low'
    dpq.restart(**restart_param)
    dpq.pause_queue()
    now_size = dpq.DataQueue().qsize()
    count = 0
    while dpq.paused_and_has_next():
        dq.get()
        count += 1
        pass
    assert count == now_size
    dpq.continue_queue()
    while dpq.has_next():
        dq.get()
        count += 1
        pass
    assert count == 100

    restart_param['device_batch'] = [1]
    restart_param['critical_process'] = 'allow_low'
    dpq.restart(**restart_param)
    count = 0
    while count < 50 and dpq.has_next():
        get = dq.get()
        assert len(get) == 1
        for k, v in enumerate(get[0]):
            assert v.datas().shape == (1, 1), print(v.datas().shape)
            pass
        count += 1
        pass

    dpq.inject_operation({'recycle': True}, device_batch=[2])
    while count < 60 and dpq.has_next():
        get = dq.get()
        assert len(get) == 1
        for k, v in enumerate(get[0]):
            assert v.datas().shape == (2, 1), print(v.datas().shape)
            pass
        count += 1
        pass

    old_size = dpq.inject_operation({'recycle': False}, device_batch=[1])
    while count < 60 + old_size and dpq.has_next():
        get = dq.get()
        assert len(get) == 1
        for k, v in enumerate(get[0]):
            assert v.datas().shape == (2, 1), print(get[0].datas().shape)
        count += 1
        pass
    assert count == 60 + old_size, print(count)
    remain_count = 100 - (50 + (10 + old_size) * 2)
    truck_count = count
    while (count - truck_count) < remain_count and dpq.has_next():
        get = dq.get()
        assert len(get) == 1
        for k, v in enumerate(get[0]):
            assert v.datas().shape == (1, 1), print(get[0].datas().shape)
        count += 1
        pass
    assert count == old_size + remain_count + 60, print(count)

    dpq.stop_generation()
    pool.join()
    print(dpq.queue_process_ret().get())
    # while dq.empty() is False or dpq.EpochDoneFlag.value is False:
    #     print('get')
    #     print(dq.get())
    pass
