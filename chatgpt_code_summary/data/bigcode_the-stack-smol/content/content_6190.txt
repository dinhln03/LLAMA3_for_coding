import copy
from enum import Enum
import multiprocessing
import numpy as np
from functools import cmp_to_key
import plotly as py
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
from collections import defaultdict
import os
from pynvml import *
import time
import matplotlib
# matplotlib.use('Agg')
import pickle
import numpy as np
from pynvml import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from matplotlib import cm
from tensorboard.plugins.hparams import keras
from line_profiler import LineProfiler
from typing import List


def get_PCIE_bandwidth():
    # if not debug_mod:
    #     PCIE_bandwidth = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_COUNT)  # KB/s => MB/ms
    #     PCIE_bandwidth /= 1000000
    # else:
    PCIE_bandwidth = 12
    return PCIE_bandwidth


GPU = int(os.environ['CUDA_VISIBLE_DEVICES'])
debug_mod = False
if not debug_mod:
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(GPU)
pyplt = py.offline.plot
PCIE_bandwidth = get_PCIE_bandwidth()
load_list = ['convolution_2d_forward_VALID', 'convolution_backward_filter_2d_VALID', 'convolution_backward_data_2d_VALID',
             'convolution_2d_forward_SAME', 'convolution_backward_filter_2d_SAME', 'convolution_backward_data_2d_SAME',
             'dropout_forward', 'dropout_backward', 'broadcast_to_NHWC',
             'broadcast_to_NCHW', 'reduce_sum_new_NHWC', 'reduce_sum_new_NCHW',
             'bn_forward_pre_activation', 'bn_backward_pre_activation', 'activation_forward_relu',
             'activation_backward_relu', 'activation_forward_softmax', 'activation_backward_softmax',
             'pooling_2d_forward_max', 'pooling_2d_backward_max', 'pooling_2d_forward_mean',
             'pooling_2d_backward_mean', 'matrix_multiply', 'matrix_elementwise_multiply_by_const', 'matrix_elementwise_add',
             'array_set', 'concat_forward', 'concat_a_backward',
             'concat_b_backward', 'sgd_update', 'cross', 'cross_backward', 'adam_mv', 'adam_compute']
optimizer_op = ['AdamOp']


class TaskType(Enum):
    swap_out = 0
    swap_in = 1


class AccessType(Enum):
    output = 0
    input = 1


class Tensor:
    def __init__(self, tensor_id, job_id, size, shape, recomputation_time, source_tensors=None, is_parameter=False, is_input_or_output=False):
        self.tensor_id = tensor_id
        self.job_id = job_id
        self.size = size
        self.swap_time = self.size / PCIE_bandwidth
        self.source_tensors = source_tensors if source_tensors is not None else []
        self.recomputation_time = recomputation_time
        self.recomputation_metric = self.size / self.recomputation_time
        self.is_parameter = is_parameter
        self.shape = shape
        if self.is_parameter or is_input_or_output:
            self.in_gpu_at_beginning = True
        else:
            self.in_gpu_at_beginning = False

    def __repr__(self):
        return f'tensor_id:{self.tensor_id}, job_id":{self.job_id}, size:{self.size}'

    def update_swap_time(self):
        PCIE_bandwidth = get_PCIE_bandwidth()
        # print(f'PCIE_bandwidth:{PCIE_bandwidth}')
        self.swap_time = self.size / PCIE_bandwidth


class TensorAccess:
    def __init__(self, tensor, time, run_time, access_type, operation_id, operation_name):
        self.tensor = tensor
        self.access_id = None
        self.start_time = None
        self.end_time = None
        self.time = time
        self.run_time = run_time
        self.access_type = access_type
        if self.access_type == AccessType.output:
            self.end_time = self.time
            self.start_time = self.time - self.run_time
        else:
            self.start_time = self.time
            self.end_time = self.time + self.run_time
        self.release_flag = False
        self.operation_id = operation_id
        self.operation_name = operation_name
        self.release_for_recomputation = []

    def to_tuple(self):
        return (self.tensor.tensor_id, self.time)

    def __repr__(self):
        return f'id={self.tensor.tensor_id}, start_time={self.start_time}, end_time={self.end_time}, time={self.time}, access_type={self.access_type}, release_flag={self.release_flag}'


class SwapTask(object):
    '''Date weighted interval'''

    def __init__(self, tensor, time, time_cost, task_type: TaskType, front_boundary=None, back_boundary=None):
        self.tensor = tensor
        self.time_cost = time_cost
        self.data_type = np.float64
        self.task_type = task_type
        self.swap_task_id = None
        assert not (front_boundary is None and back_boundary is None)
        # 最早开始时间
        self.front_boundary = front_boundary
        # 最晚结束时间
        self.back_boundary = back_boundary
        self.time = time
        self.execute_time = None
        self.execute_ref = None
        self.start_time_ = None
        self.end_time_ = None

    @property
    def start_time(self):
        return self.start_time_

    @start_time.setter
    def start_time(self, value):
        self.start_time_ = value
        if self.task_type == TaskType.swap_out:
            self.time = self.start_time_

    @property
    def end_time(self):
        return self.end_time_

    @end_time.setter
    def end_time(self, value):
        self.end_time_ = value
        if self.task_type == TaskType.swap_in:
            self.time = self.end_time_

    @classmethod
    def from_access(cls, access: TensorAccess, weight, task_type, front_boundary=None, back_boundary=None):
        return cls(access.tensor, weight, access.time, access.tensor.swap_time, task_type, front_boundary=front_boundary, back_boundary=back_boundary)

    def __repr__(self):
        return f'id={self.tensor}, type={self.task_type}, start_time={self.start_time}, end_time={self.end_time}, time={self.time}'


def numpy_ewma_vectorized(data, window):
    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha
    n = data.shape[0]

    pows = alpha_rev ** (np.arange(n + 1))

    scale_arr = 1 / pows[:-1]
    offset = data[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n - 1)

    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


debug_num = 0


def create_model(n):
    model = Sequential()
    model.add(Dense(units=2048, activation='tanh', input_dim=n))
    model.add(Dense(units=2048, activation='tanh'))
    model.add(Dense(units=1, activation='relu'))
    return model


def load(opname, n):
    model = create_model(n)
    model.load_weights('model_parameter/' + opname + '_model.hdf5', by_name=True, skip_mismatch=True)
    return model


def get_predicted_execution_time(op_name, inputs_of_model, logged_time: list):
    return logged_time[0]


def liveness_analysis(tensor_access_list):
    global tensor_access_by_tensor
    # 活跃性分析结果生成
    for job_id in range(len(tensor_access_list)):
        tmp = set()
        for i in range(len(tensor_access_list[job_id]) - 1, -1, -1):
            tensor_access = tensor_access_list[job_id][i]
            accesses_of_tensor = tensor_access_by_tensor[tensor_access.tensor.job_id][tensor_access.tensor]
            if tensor_access.tensor not in tmp and len(accesses_of_tensor) > 1 and tensor_access == accesses_of_tensor[-1]:
                # 参数不会释放
                if not tensor_access.tensor.is_parameter:
                    tmp.add(tensor_access.tensor)
                    tensor_access.release_flag = True


def is_overlap(task: SwapTask, target: SwapTask):
    return task != target and (
            target.start_time < task.end_time < target.end_time or target.start_time < task.start_time < target.end_time or task.start_time < target.end_time < task.end_time or task.start_time < target.start_time < task.end_time)


def get_free_intervals(target_task, swap_schedule, access_of_target_tensor, key=0, asc=True):
    target_task.tensor.update_swap_time()
    # 列出在可行区间内的所有空白时间区间，并按区间排序
    if target_task.back_boundary - target_task.front_boundary < target_task.time_cost:
        return []
    intervals = []
    for task in swap_schedule:
        # if target_task.back_boundary < task.start_time:
        #     continue
        # elif task.end_time < target_task.front_boundary:
        #     break
        if target_task.front_boundary <= task.start_time < task.end_time <= target_task.back_boundary:
            intervals.append((task.start_time, task.end_time))
        elif task.start_time < target_task.front_boundary < task.end_time < target_task.back_boundary:
            intervals.append((target_task.front_boundary, task.end_time))
        elif target_task.front_boundary < task.start_time < target_task.back_boundary < task.end_time:
            intervals.append((task.start_time, target_task.back_boundary))
        elif task.start_time < target_task.front_boundary < target_task.back_boundary < task.end_time:
            return []
    intervals = sorted(intervals, key=lambda x: x[0])
    # 区间融合，确保区间之间无交集
    occupied_intervals = []
    i = 0
    while i < len(intervals):
        interval = intervals[i]
        l = interval[0]
        r = interval[1]
        flag = False
        while i < len(intervals) - 1 and intervals[i + 1][0] <= r:
            r = max(r, intervals[i + 1][1])
            flag = True
            i += 1
        occupied_intervals.append((l, r))
        if not flag:
            i += 1
    not_occupied_intervals = []
    s = target_task.front_boundary
    for interval in occupied_intervals:
        if s < interval[0]:
            not_occupied_intervals.append((s, interval[0]))
        s = interval[1]
    if s < target_task.back_boundary:
        not_occupied_intervals.append((s, target_task.back_boundary))
    if len(not_occupied_intervals) == 0:
        return []
    i = 0
    j = 0
    # 按照区间起点排序
    not_occupied_intervals = sorted(not_occupied_intervals, key=lambda x: x[key], reverse=False)
    # 防止区间与被调度张量的access重合
    while j < len(access_of_target_tensor):
        if i >= len(not_occupied_intervals):
            break
        access = access_of_target_tensor[j]
        start, end = not_occupied_intervals[i]
        if start < access.start_time < end <= access.end_time:
            not_occupied_intervals[i] = (start, access.start_time)
            i += 1
        elif start < access.start_time < access.end_time < end:
            not_occupied_intervals[i] = (start, access.start_time)
            not_occupied_intervals.insert(i + 1, (access.end_time, end))
            i += 1
            j += 1
        elif start == access.start_time < end < access.end_time:
            not_occupied_intervals.pop(i)
            j += 1
        elif access.start_time <= start < access.end_time < end:
            not_occupied_intervals[i] = (access.end_time, end)
            j += 1
        elif access.start_time <= start < end <= access.end_time:
            not_occupied_intervals.pop(i)
        else:
            j += 1
    # 按照区间终点排序
    if not asc:
        not_occupied_intervals = sorted(not_occupied_intervals, key=lambda x: x[key], reverse=not asc)
    return not_occupied_intervals


def generate_swap_recomputation_release_order(tensor_access_by_tensor, swap_scheduler, recomputations, job_num):
    swap_orders = defaultdict(list)
    release_orders = defaultdict(list)
    recomp_orders = defaultdict(list)
    for job_id in range(job_num):
        # 按id排序
        tensor_accesses = sorted([i for tmp in tensor_access_by_tensor[job_id].values() for i in tmp], key=lambda x: x.tensor.tensor_id)
        # 按起始时间排序
        swap_tasks = sorted(swap_scheduler[job_id], key=lambda x: x.start_time)
        for i in range(len(swap_tasks)):
            swap_tasks[i].swap_task_id = i
        releases = []
        swaps = []
        recomps = []
        for access in tensor_accesses:
            if access.release_flag:
                releases.append((access.operation_id, access.tensor.tensor_id))
        release_orders[job_id] = releases
        for access in recomputations:
            recomps.append((access.operation_id, access.tensor.tensor_id, access.release_for_recomputation))
        recomp_orders[job_id] = recomps
        for task in swap_tasks:
            # if task.task_type==TaskType.swap_out:
            # (task_id, node_id(tensor_id), start_time, start_node, move_to_gpu, start_node_type)
            ref = task.execute_ref.operation_id
            swaps.append([task.tensor.tensor_id, task.execute_time, ref, 0 if task.task_type == TaskType.swap_out else 1, 1, task.start_time])
        swap_orders[job_id] = list(map(lambda x: x[:-1], sorted(swaps, key=lambda x: x[-1])))
    return release_orders, swap_orders, recomp_orders


def draw_all_task(tensor_access_by_tensor, swap_scheduler, job_num):
    for job_id in range(job_num):
        tmp = list(tensor_access_by_tensor[job_id].values())
        res = []
        for sub_list in tmp:
            res.extend(sub_list)
        draw(sorted(res, key=lambda x: x.start_time), swap_scheduler[job_id])


class MemoryAnalyzer:
    def __init__(self, tensor_access_list, tensors):
        self.tensor_access_list = tensor_access_list
        self.tensors = tensors
        self.next_swap_tasks_index = 0

    def insert_sort(self, list_with_order: list, list_b: list, cmp):
        # 升序
        for obj_b in list_b:
            i = 0
            mid = 0
            j = len(list_with_order) - 1
            while i < j:
                mid = (i + j) // 2
                obj_mid = list_with_order[mid]
                flag = cmp(obj_mid, obj_b)
                if flag == -1:
                    # mid<b
                    if mid == i:
                        # i=mid<=j, mid<b, 比较b和j
                        flag2 = cmp(list_with_order[j], obj_b)
                        if flag2 == -1:
                            # i=mid<=j<b, 插入位置在j+1
                            mid = j
                        elif flag2 == 1:
                            # i=mid<b<j, 插入位置在j
                            mid = j - 1
                        else:
                            # i=mid<=j=b, 插入位置在j+1
                            mid = j
                        break
                    i = mid
                elif flag == 1:
                    # b<mid
                    if mid == j:
                        # i<=mid=j, b<mid, 比较i和b
                        flag2 = cmp(list_with_order[i], obj_b)
                        if flag2 == -1:
                            # i<b<mid=j, 插入位置在i+1
                            mid = i
                        elif flag2 == 1:
                            # b<i<mid=j, 插入位置在i
                            mid = i - 1
                        else:
                            # i=b<mid=j, 插入位置在i+1
                            mid = i
                        break
                    j = mid
                elif flag == 0:
                    # b==mid，插入位置在mid+1
                    break
            list_with_order.insert(mid + 1, obj_b)
        return list_with_order

    def custom_cmp(self, x, y):
        if x.time < y.time:
            return -1
        elif x.time > y.time:
            return 1
        else:
            if x.start_time < y.start_time:
                return -1
            elif x.start_time > y.start_time:
                return 1
            else:
                # if isinstance(x,TensorAccess) and isinstance(y, SwapTask):
                #     return 1
                # elif isinstance(x, SwapTask) and isinstance(y, TensorAccess):
                #     return -1
                return 0

    def custom_cmp_end_time(self, x, y):
        if x.end_time < y.end_time:
            return -1
        elif x.end_time > y.end_time:
            return 1
        else:
            return 0

    def get_max_memory_used(self, swap_tasks, swapped_out_tensor):
        delta = len(swap_tasks)
        if self.next_swap_tasks_index == 0:
            # 初始化时间轴
            tmp = copy.copy(self.tensor_access_list)
            tmp.extend(swap_tasks)
            self.time_axis = sorted(tmp, key=cmp_to_key(self.custom_cmp))
            self.end_time_axis = sorted(copy.copy(tmp), key=cmp_to_key(self.custom_cmp_end_time))
            # self.last_unused_swap_tasks = copy.copy(swap_tasks)
        else:
            # 更新时间轴
            # assert swap_tasks[:self.next_swap_tasks_index] == self.last_unused_swap_tasks
            # self.last_unused_swap_tasks = copy.copy(swap_tasks)
            swap_tasks = swap_tasks[self.next_swap_tasks_index:]
            self.time_axis = self.insert_sort(self.time_axis, swap_tasks, self.custom_cmp)
            self.end_time_axis = self.insert_sort(self.end_time_axis, swap_tasks, self.custom_cmp_end_time)
        self.index_of_end_time_axis = {self.end_time_axis[i]: i for i in range(len(self.end_time_axis))}
        # 计算显存开销
        # occupied by handle, cudnn, cuda stream and cudart
        memory_used = 0
        max_memory_actual = float('-inf')
        in_gpu_tensors = set()
        max_memory_tensors = set()
        last_input_tensor_access = None
        max_last_access = None
        wait_to_be_released = []
        max_time = None
        # foot_print = {}
        # 首先把输入的x，y以及所有没被swap out的参数载入显存，因为他们从上轮迭代结束时就一直在显存里面
        for tensor in self.tensors:
            if tensor.in_gpu_at_beginning and tensor not in swapped_out_tensor:
                in_gpu_tensors.add(tensor)
                memory_used += tensor.size
        for time_index, event in enumerate(self.time_axis):
            i = len(wait_to_be_released) - 1
            while i >= 0:
                access = wait_to_be_released[i]
                # 如果此刻时间已经过了释放时间，则释放该访问的附带影响
                if event.time >= access.end_time:
                    wait_to_be_released.pop(i)
                    memory_used -= access.tensor.size
                    in_gpu_tensors.remove(access.tensor)
                i -= 1
            if isinstance(event, TensorAccess):
                if event.access_type == AccessType.output:
                    if event.tensor not in in_gpu_tensors:
                        # 新参数不额外占用空间
                        if event.operation_name not in optimizer_op:
                            memory_used += event.tensor.size
                        in_gpu_tensors.add(event.tensor)
                else:
                    # 用完即释放的
                    # input本身并不增加gpu使用，swap in增加
                    if event.release_flag:
                        wait_to_be_released.append(event)
                    else:
                        last_input_tensor_access = event
            elif isinstance(event, SwapTask):
                # 使用按照结束时间排序的时间轴进行倒序查找
                last_event = None
                # idx = end_time_axis.index(event)
                idx = self.index_of_end_time_axis[event]
                for j in range(idx - 1, -1, -1):
                    if isinstance(self.end_time_axis[j], TensorAccess) and self.end_time_axis[j].end_time <= event.start_time:
                        last_event = self.end_time_axis[j]
                        break
                if last_event is None:
                    last_event = self.tensor_access_list[0]
                event.execute_ref = last_event
                event.execute_time = event.start_time - last_event.end_time
                if event.task_type == TaskType.swap_in:
                    memory_used += event.tensor.size
                    in_gpu_tensors.add(event.tensor)
                else:
                    memory_used -= event.tensor.size
                    in_gpu_tensors.remove(event.tensor)
            # foot_print[time] = memory_used
            if memory_used > max_memory_actual:
                # max_memory_actual与是否有考虑价值无关，单纯计量峰值
                max_memory_actual = memory_used
                max_memory_tensors = copy.copy(in_gpu_tensors)
                max_last_access = last_input_tensor_access
                max_time = event.time
        self.next_swap_tasks_index = delta
        return max_memory_actual, max_memory_tensors, max_last_access, max_time, self.time_axis


def run_global_memory_analysis(swap_tasks, swapped_out_tensor):
    global job_num
    global global_memory_analyzer
    max_memory = 0
    max_memory_tensors = []
    last_input_accesses = []
    max_time = []
    # foot_prints = []
    time_axis = []
    for job_id in range(job_num):
        job_max_memory, job_max_memory_tensors, last_input_access, now_time, t_axis = global_memory_analyzer[job_id].get_max_memory_used(swap_tasks[job_id], swapped_out_tensor)
        time_axis.append(t_axis)
        # foot_prints.append(foot_print)
        max_memory_tensors.extend(job_max_memory_tensors)
        last_input_accesses.append(last_input_access)
        max_time.append(now_time)
        max_memory += job_max_memory
    return max_memory, max_memory_tensors, last_input_accesses, max_time, time_axis


def draw(tensor_access_list, swap_schedule):
    df = []
    id_color = {'OTA': 'rgb(255, 0, 102)', 'ITA': 'rgb(68, 114, 196)', 'Swap In': 'rgb(237, 137, 69)', 'Swap Out': 'rgb(112, 173, 71)'}
    for tensor_access in tensor_access_list:
        # input 蓝色，output红色
        df.append(dict(Task=f'tensor_id:{tensor_access.tensor.tensor_id}, size:{tensor_access.tensor.size}', Start=tensor_access.start_time, Finish=tensor_access.end_time,
                       Resource='OTA' if tensor_access.access_type == AccessType.output else 'ITA'))
    for task in swap_schedule:
        df.append(dict(Task=f'tensor_id:{task.tensor.tensor_id}, size:{task.tensor.size}', Start=task.start_time, Finish=task.end_time, Resource='Swap In' if task.task_type == TaskType.swap_in else 'Swap Out'))

    fig = ff.create_gantt(df, colors=id_color, index_col='Resource', group_tasks=True, show_colorbar=True, showgrid_x=True, showgrid_y=True, title=f'ratio={ratio}')
    fig['layout']['xaxis'].update({'type': None})
    fig.update_layout(
        height=900,
        width=1600,
    )
    pyplt(fig, filename=f'../../pic/job{tensor_access_list[0].tensor.job_id}.html', auto_open=True)


def try_swap_in(swap_in_task: SwapTask, swap_scheduler, access_of_target_tensor):
    # swap_in越晚越好，按结束时间降序排序
    free_intervals = get_free_intervals(swap_in_task, swap_scheduler[swap_in_task.tensor.job_id], access_of_target_tensor, 1, asc=False)
    succeed = False
    for interval in free_intervals:
        if interval[1] - interval[0] >= swap_in_task.time_cost:
            swap_in_task.end_time = interval[1]
            swap_in_task.start_time = swap_in_task.end_time - swap_in_task.time_cost
            swap_scheduler[swap_in_task.tensor.job_id].append(swap_in_task)
            succeed = True
            break
    if not succeed:
        return False
    else:
        return True


def can_next_input_access_swap_in(i, all_access_of_tensor, swap_out_task, swap_scheduler):
    # 至少将第一个访问swap in才算成功，后续的能换入的话，则把前一个的release_flag设为True
    access = all_access_of_tensor[i]
    swap_in_task = SwapTask(access.tensor, access.time, access.tensor.swap_time, TaskType.swap_in,
                            front_boundary=swap_out_task.end_time if swap_out_task.end_time > all_access_of_tensor[i - 1].end_time else all_access_of_tensor[i - 1].end_time,
                            back_boundary=access.time)
    return try_swap_in(swap_in_task, swap_scheduler, tensor_access_by_tensor[swap_in_task.tensor.job_id][swap_in_task.tensor])


def get_framework_info(info, logged_time, job_id):
    global global_tensors
    tensors = {}
    tensor_access_list = []
    global_time = 0
    parameter = []
    # tensor_id: execution time of operator which generate the tensor
    operator_execution_time = []
    # for output_tensor_id, input_tensor_id, output_tensor_size, operation_name, is_parameter, shape, inputs_of_model in info:
    for tensor_info, input_tensor_id, operation_name, operation_id, is_parameter, inputs_of_model, _ in info:
        # is_parameter: 生成的张量是否为参数
        # 输入的为Byte
        # 转换为MB
        input_tensors = []
        for tensor_id in input_tensor_id:
            input_tensor = tensors[tensor_id]
            input_tensors.append(input_tensor)
        time_cost = get_predicted_execution_time(operation_name, inputs_of_model, logged_time[operation_id])
        for output_tensor_id, output_tensor_size, shape in tensor_info:
            output_tensor_size = output_tensor_size / 1000000
            operator_execution_time.append(time_cost)
            if operation_name in optimizer_op:
                is_parameter = 1
            output_tensor = Tensor(tensor_id=output_tensor_id, job_id=job_id, size=output_tensor_size, source_tensors=input_tensors, recomputation_time=time_cost, is_parameter=is_parameter, shape=shape)
            output_access = TensorAccess(tensor=output_tensor, time=global_time + time_cost, run_time=time_cost, access_type=AccessType.output, operation_id=operation_id, operation_name=operation_name)
            tensor_access_list.append(output_access)
            tensors[output_tensor.tensor_id] = output_tensor
            if is_parameter:
                parameter.append(output_tensor)
        for tensor_id in input_tensor_id:
            input_tensor = tensors[tensor_id]
            input_access = TensorAccess(tensor=input_tensor, time=global_time, run_time=time_cost, access_type=AccessType.input, operation_id=operation_id, operation_name=operation_name)
            tensor_access_list.append(input_access)
        global_time += time_cost

    tensors = list(tensors.values())
    global_tensors[job_id] = tensors
    tensor_access_list = sorted(tensor_access_list, key=lambda x: x.time)
    dic = defaultdict(list)
    for access in tensor_access_list:
        dic[access.tensor].append(access)
    for k, v in dic.items():
        dic[k] = sorted(v, key=lambda x: x.time)
    tensor_access_by_tensor[job_id] = dic

    swap_scheduler = []
    # 对参数进行swap in调度
    # earliest_swap = None
    # earliest_time = float('inf')
    # 从最早的参数开始安排
    parameter = sorted(parameter, key=lambda x: dic[x][0].start_time)
    return tensor_access_list, swap_scheduler, parameter, operator_execution_time


# 随机生成数据用的参数
times = 150
tensors = 50
time_scale = times
ratio = 1

# 全局变量
job_num = 0
global_tensor_access = [[]]
tensor_access_by_tensor = []
weight = 1
jobs_weights = []
# jobs_weight = [1, 1, 1, 1, 1]
total_memory = 0
enable_recomputation = True
global_graphs = []
global_tensors = {}
swap_scheduler = []
parameters = []
models = {}
global_memory_analyzer = []


# load_all_model()


def init(logged_times: list, gpu: int):
    global job_num
    global global_tensor_access
    global tensor_access_by_tensor
    global total_memory
    global handle
    global jobs_weights
    global global_graphs
    global global_tensors
    global swap_scheduler
    global parameters
    global global_memory_analyzer
    global_tensor_access = [[]]
    tensor_access_by_tensor = []
    global_tensors = {}
    swap_scheduler = []
    parameters = []
    global_memory_analyzer = []
    graphs = global_graphs
    jobs_weights = [weight for _ in range(len(graphs))]
    tensor_access_by_tensor = [[] for _ in range(job_num)]
    # 获取当前剩余显存总量
    if not debug_mod:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(gpu)
        total_memory = nvmlDeviceGetMemoryInfo(handle).free / 1000000
    else:
        total_memory = 6000
    job_num = len(graphs)
    tmp = [get_framework_info(graphs[i], logged_times[i], i) for i in range(job_num)]
    global_tensor_access = [tmp[i][0] for i in range(job_num)]
    swap_scheduler = [tmp[i][1] for i in range(job_num)]
    parameters = [tmp[i][2] for i in range(job_num)]
    for i in range(job_num):
        global_memory_analyzer.append(MemoryAnalyzer(global_tensor_access[i], global_tensors[i]))


def add_job(graph, job_id, gpu: int):
    global global_graphs
    assert job_id == len(global_graphs) or global_graphs[job_id] is None
    if job_id == len(global_graphs):
        global_graphs.append(graph)
    else:
        global_graphs[job_id] = graph
    init([[] for _ in range(job_num)], gpu)


def remove_job(job_id, gpu: int):
    global global_graphs
    global_graphs[job_id] = None
    init([], gpu)


def generate_scheduling_plan(logged_times, gpu: int):
    # 如果是此时logged_times已经清空，则
    # logged_times: [[(operation_id, [time, time, time])]]，外层索引为job_id
    global total_memory
    global global_tensors
    init(logged_times, gpu)
    # 指数加权平均更新估计时间
    tensor_nums = list(map(lambda x: len(x), tensor_access_by_tensor))
    swap_out_number_limits = [int(weight * tensor_num) for weight, tensor_num in zip(jobs_weights, tensor_nums)]
    swap_out_number = [0 for _ in tensor_nums]
    swapped_out_tensor = set()
    swapped_in_source_tensor = set()
    swap_out_dict = {}
    swapped_in_access = set()
    recomputations = []
    recomputation_tensor = set()
    # key：tensor，value：[所有释放这个张量的重计算对应的在recomputations中的index]
    # 上一轮没有成功的swap_out时为False
    swapped_flag = True
    recomputation_flag = True
    iter = 0
    original_memory_used = 0
    last_memory_used = 0
    job_id_ordered_by_weights = list(map(lambda x: x[0], sorted([(job_id, weights) for job_id, weights in enumerate(jobs_weights)], key=lambda x: x[1], reverse=True)))
    max_memory_footprint = []
    # draw_all_task(tensor_access_by_tensor, swap_scheduler, job_num)
    while swapped_flag or (recomputation_flag and enable_recomputation):
        # MB
        if not debug_mod:
            total_memory = nvmlDeviceGetMemoryInfo(handle).free / 1000000
        else:
            total_memory = 6000
        max_memory, max_tensors, last_input_accesses, max_time, time_axis = run_global_memory_analysis(swap_scheduler, swapped_out_tensor)
        max_memory_footprint.append(max_memory)
        # 最后三次迭代的峰值，做一阶差分，结果的最大值大于上一次峰值的0.05%以上或迭代次数小于100轮才继续~`
        if len(max_memory_footprint) > 3 and max([max_memory_footprint[i] - max_memory_footprint[i + 1] for i in range(len(max_memory_footprint) - 3, len(max_memory_footprint) - 1)]) < max_memory_footprint[
            -1] * 0.0005 and iter > 100:
            break
        if iter == 0:
            original_memory_used = max_memory
            liveness_analysis(global_tensor_access)
        else:
            last_memory_used = max_memory
        # print(f'iter:{iter}, max_memory:{max_memory}')
        max_tensors = sorted(max_tensors, key=lambda x: x.size, reverse=True)
        if swapped_flag:
            swapped_flag = False
            for tensor in max_tensors:
                # 对该张量进行swap_out计划的安排
                is_new_parameter = tensor.is_parameter and tensor_access_by_tensor[tensor.job_id][tensor][0].operation_name in optimizer_op and len(tensor_access_by_tensor[tensor.job_id][tensor]) == 1
                if not is_new_parameter:
                    if swap_out_number[tensor.job_id] <= swap_out_number_limits[tensor.job_id] and len(tensor_access_by_tensor[tensor.job_id][tensor]) > 1:
                        # swapped_out表示所有可能的swap_in已经调度过了
                        if tensor not in swapped_out_tensor:
                            all_access_of_tensor = tensor_access_by_tensor[tensor.job_id][tensor][1:]
                            # 首先确定swap_out的时间范围，最迟不能超过此时此刻，最早不能超过第一次访问结束时刻
                            output_access = tensor_access_by_tensor[tensor.job_id][tensor][0]
                            assert output_access.access_type == AccessType.output
                            if last_input_accesses[tensor.job_id] is not None:
                                # 此时此刻
                                back_boundary = last_input_accesses[tensor.job_id].time
                            else:
                                last_time_access = tensor_access_by_tensor[tensor.job_id][tensor][-1]
                                back_boundary = last_time_access.time + tensor.swap_time
                            succeed = False
                            front_boundary = output_access.time
                            # failed_input_access = []
                            swap_out_succeed = True
                            have_next_ITA = True
                            # 如果是因为swap out放不下，则不用继续更新可行区间了，直接break
                            while not succeed and front_boundary < back_boundary and swap_out_succeed and have_next_ITA:
                                swap_out_task = SwapTask(tensor, output_access.time, tensor.swap_time, TaskType.swap_out, front_boundary=front_boundary, back_boundary=back_boundary)
                                free_intervals = get_free_intervals(swap_out_task, swap_scheduler[swap_out_task.tensor.job_id], tensor_access_by_tensor[tensor.job_id][tensor])
                                selected_first_access_index = None
                                # 选出能容纳该任务的剩余空间
                                swap_out_succeed = False
                                have_next_ITA = False
                                for interval in free_intervals:
                                    if interval[1] - interval[0] >= swap_out_task.time_cost:
                                        swap_out_succeed = True
                                        swap_out_task.start_time = interval[0]
                                        swap_out_task.end_time = swap_out_task.start_time + swap_out_task.time_cost
                                        swap_scheduler[swap_out_task.tensor.job_id].append(swap_out_task)
                                        # 看一下后面第一个swap_in能否放下
                                        for i, access in enumerate(all_access_of_tensor):
                                            # 找到后面第一个访问
                                            if access.start_time >= swap_out_task.end_time:
                                                have_next_ITA = True
                                                if can_next_input_access_swap_in(i, all_access_of_tensor, swap_out_task, swap_scheduler):
                                                    swapped_out_tensor.add(tensor)
                                                    swap_out_dict[tensor] = swap_out_task
                                                    swapped_in_access.add(access)
                                                    swap_out_number[tensor.job_id] += 1
                                                    selected_first_access_index = i
                                                    succeed = True
                                                    swapped_flag = True
                                                else:
                                                    # failed_input_access.append(access)
                                                    swap_scheduler[swap_out_task.tensor.job_id].remove(swap_out_task)
                                                    # 修正swap_out_task前向限制为这个失败的input_access的结束时间
                                                    front_boundary = access.end_time
                                                    assert tensor not in swapped_out_tensor
                                                    # swapped_out_tensor.remove(tensor)
                                                break
                                        if not succeed:
                                            if swap_out_task in swap_scheduler[swap_out_task.tensor.job_id]:
                                                swap_scheduler[swap_out_task.tensor.job_id].remove(swap_out_task)
                                                # 如果不是因为swap out没安排下则重新生成区间
                                                break
                                        else:
                                            break
                            # 安排失败
                            if not succeed:
                                continue
                            if not is_new_parameter:
                                # 后续的能换入的话，则把前一个的release_flag设为True
                                for i in range(selected_first_access_index + 1, len(all_access_of_tensor)):
                                    access = all_access_of_tensor[i]
                                    if i == 0 or access in swapped_in_access:
                                        continue
                                    else:
                                        if can_next_input_access_swap_in(i, all_access_of_tensor, swap_out_task, swap_scheduler):
                                            # print(f'成功{access}')
                                            swapped_in_access.add(access)
                                            if all_access_of_tensor[i - 1].start_time > swap_out_task.end_time:
                                                all_access_of_tensor[i - 1].release_flag = True
                            if swapped_flag:
                                break
                # 如果是新参数，则尝试对新参数进行swap out，对对应的旧参数进行swap in
                else:
                    if tensor not in swapped_out_tensor:
                        output_access = tensor_access_by_tensor[tensor.job_id][tensor][0]
                        assert output_access.access_type == AccessType.output
                        swap_out_task = SwapTask(tensor, time=output_access.time, time_cost=tensor.swap_time, task_type=TaskType.swap_out, front_boundary=output_access.end_time, back_boundary=float('inf'))
                        free_intervals = get_free_intervals(swap_out_task, swap_scheduler[swap_out_task.tensor.job_id], tensor_access_by_tensor[tensor.job_id][tensor])
                        for interval in free_intervals:
                            if interval[1] - interval[0] >= swap_out_task.time_cost:
                                swap_out_task.start_time = interval[0]
                                swap_out_task.end_time = swap_out_task.start_time + swap_out_task.time_cost
                                swap_scheduler[swap_out_task.tensor.job_id].append(swap_out_task)
                                # 找到对应的旧参数张量
                                # 由于二者可行域无关，所以直接查看对应的swap in 能否调度
                                for t in tensor.source_tensors:
                                    if t.is_parameter and t not in swapped_in_source_tensor:
                                        # 试图swap in
                                        # 找到第一次input访问(feed_dict不实际使用)
                                        first_access = tensor_access_by_tensor[t.job_id][t][1]
                                        assert first_access.access_type == AccessType.input
                                        swap_in_task = SwapTask(t, first_access.time, first_access.tensor.swap_time, TaskType.swap_in, front_boundary=0, back_boundary=first_access.start_time)
                                        res = try_swap_in(swap_in_task, swap_scheduler, tensor_access_by_tensor[t.job_id][t])
                                        # assert not res, f'swap in parameter:{t} failed'
                                        if res:
                                            swapped_in_source_tensor.add(t)
                                            swapped_out_tensor.add(tensor)
                                            swap_out_dict[tensor] = swap_out_task
                                            swapped_in_access.add(first_access)
                                            swap_out_number[tensor.job_id] += 1
                                            swapped_flag = True
                                        else:
                                            swap_scheduler[swap_out_task.tensor.job_id].remove(swap_out_task)
                                            assert tensor not in swapped_out_tensor
                                        break
                                break
        elif enable_recomputation:
            recomputation_flag = False
            # 需要重计算
            if max_memory >= total_memory:
                for job_id in job_id_ordered_by_weights:
                    max_tensors_filtered = []
                    for tensor in max_tensors:
                        # 张量不是参数，没被逐出过，且他的所有源张量从未被swap或recomputation
                        if not tensor.is_parameter and tensor not in swapped_out_tensor and tensor.source_tensors is not None and len(tensor.source_tensors) > 0 and \
                                False not in [t not in swapped_out_tensor for t in tensor.source_tensors] and False not in [t not in recomputations for t in tensor.source_tensors]:
                            max_tensors_filtered.append(tensor)
                    if len(max_tensors_filtered) == 0:
                        continue
                    max_tensors_by_metric = sorted(max_tensors_filtered, key=lambda x: x.recomputation_metric, reverse=True)
                    # 选取metric最大的张量
                    tensor = max_tensors_by_metric[0]
                    # 找到此刻对应的下一个访问
                    now_time = max_time[job_id]
                    all_access_of_tensor = tensor_access_by_tensor[tensor.job_id][tensor]
                    for i, access in enumerate(all_access_of_tensor):
                        if access.access_type == AccessType.input and access not in recomputations:
                            if access.start_time >= now_time:
                                for source_tensor in access.tensor.source_tensors:
                                    accesses = tensor_access_by_tensor[source_tensor.job_id][source_tensor]
                                    for temp_acc in accesses:
                                        # 　确保source被release过的不进行重计算
                                        if temp_acc.release_flag and temp_acc.end_time <= access.start_time:
                                            break
                                    else:
                                        recomputations.append(access)
                                        all_access_of_tensor[i - 1].release_flag = True
                                        recomputation_flag = True
                                        recomputation_tensor.add(access.tensor)
                                    break
                            break
        iter += 1
    # fig = go.Figure(data=[go.Scatter(x=list(original_memory_footprint[0].keys()), y=list(original_memory_footprint[0].values())), go.Scatter(x=list(foot_prints[0].keys()), y=list(foot_prints[0].values()))])
    # plotly.offline.plot(fig, filename='../../pic/footprint.html')
    # if not debug_mod:
    #     total_memory = nvmlDeviceGetMemoryInfo(handle).free / 1000000
    # else:
    #     total_memory = 6000
    # stats = 'succeed' if max_memory < total_memory else ' failure'
    # print(f'scheduling {stats}')
    # draw_all_task(tensor_access_by_tensor, swap_scheduler, job_num)
    memory_saved_ratio = format((1 - last_memory_used / original_memory_used) * 100, '.2f')
    print(f'memory_saved_ratio:{memory_saved_ratio}%')
    print(f'swap ratio:{len(swap_scheduler[0]) / len(global_tensors)}')
    # print(f'recomputations:{recomputations}')
    return generate_swap_recomputation_release_order(tensor_access_by_tensor, swap_scheduler, recomputations, job_num)


def multiprocess_init(global_message_queue: multiprocessing.Queue, global_control_queue: multiprocessing.Queue, total_job_number):
    # swap_order = [(20, 0, 20, 0)]
    # control_messages = []
    # control_message = [swap_order, [], []]
    # control_messages.append(control_message)
    # global_control_queue.put(control_messages)
    logged_times = []
    log_repeat = 0
    alpha = 0.9
    second_schedule_finished = False

    # todo 设置从executor到algorithm的job_id的映射
    map_out_to_in = {}
    map_in_to_out = {}
    global job_num
    job_num = 0

    while True:
        if not global_message_queue.empty():
            global_message = global_message_queue.get()
            job_id = global_message[0]
            message_type = global_message[1][0]
            message_graph = global_message[1][1]

            if message_type == 0:

                # print("job_id =", job_id)

                job_num += 1
                map_out_to_in[job_id] = job_num - 1
                map_in_to_out[job_num - 1] = job_id
                job_id_in = job_num - 1

                logged_times.append([])
                global_graphs.append(message_graph)
                tensor_num = len(message_graph)

                # with open("../../global_graphs", "wb") as f1:
                #     pickle.dump(global_graphs, f1)

                for i in range(tensor_num):
                    # print(message_graph[i][6])
                    logged_times[job_id_in].append([message_graph[i][6]])

                s = time.time()
                if job_num == total_job_number:
                    release_order, swap_order, recomputation_order = generate_scheduling_plan(logged_times, 0)
                    print(f'time:{time.time() - s}')
                    control_messages = {}
                    for i in range(job_num):
                        # print(swap_order)
                        control_message = [swap_order[i], release_order[i], recomputation_order[i]]
                        control_messages[map_in_to_out[i]] = control_message
                    global_control_queue.put(control_messages)
            else:

                job_id_in = map_out_to_in[job_id]

                total_time_old = 0
                for run_time in logged_times[job_id_in]:
                    total_time_old += run_time[0]
                total_time_new = 0
                for run_time in message_graph:
                    total_time_new += run_time[1]
                change_rate = abs(total_time_new - total_time_old) / total_time_old
                print("change rate is ", change_rate)
                # print("total time new is", total_time_new)
                # print("total time old is", total_time_old)

                if change_rate > 0.3:
                    is_replan = True
                else:
                    is_replan = False

                # with open("./log/total_time.txt", "a") as f1:
                #     print(total_time_new, file=f1)

                # todo 此处控制了在一定轮数之后才进行决策
                log_repeat += 1
                if log_repeat > 0 and (is_replan or (not second_schedule_finished)):

                    second_schedule_finished = True
                    # with open("../../logged_times", "wb") as f1:
                    #     pickle.dump(logged_times, f1)

                    for node_message in message_graph:
                        time_new = node_message[1] * alpha + logged_times[job_id_in][node_message[0]][0] * (1 - alpha)
                        logged_times[job_id_in][node_message[0]] = [time_new]

                    release_order, swap_order, recomputation_order = generate_scheduling_plan(logged_times, 0)

                    print(logged_times)

                    control_messages = {}

                    for i in range(job_num):
                        print(swap_order)
                        control_message = [swap_order[i], release_order[i], recomputation_order[i]]
                        control_messages[map_in_to_out[i]] = control_message
                    global_control_queue.put(control_messages)
                # print(logged_times[0])


if debug_mod and __name__ == '__main__':
    import pickle

    with open('../../global_graphs', 'rb') as f:
        g = pickle.load(f)
    global_graphs = g
    with open('../../logged_times', 'rb') as f:
        logged_times = pickle.load(f)
    job_num = 1
    # profiler = LineProfiler()
    # profiler.add_function(get_free_intervals)
    # # profiler.add_function(get_occupied_intervals)
    # # profiler.add_function(MemoryAnalyzer.get_max_memory_used)
    # # profiler.add_function(run_global_memory_analysis)
    # profiler_wrapper = profiler(generate_scheduling_plan)
    # res = profiler_wrapper(logged_times, 0)
    # profiler.print_stats()
    release_order, swap_order, recomputation_order = generate_scheduling_plan(logged_times, 0)
