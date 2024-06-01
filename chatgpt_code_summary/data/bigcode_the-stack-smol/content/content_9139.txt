#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a video or zmq with a certain extension
(e.g., .jpg) in a folder. Sample: 
python tools/infer_from_video.py \
--cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
--output-dir ./output \
--image-ext jpg \
--wts generalized_rcnn/model_final.pkl \
--video ~/data/video3.h264
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import sys
import time
import zmq
import numpy as np
import os

from caffe2.python import workspace
import glob
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
# import arp.line_detection as detection
from multiprocessing import Process, Queue
from Queue import Empty
import json
import math
import copy
import arp.const as const
from arp.fusion_kalman import Fusion
from arp.fusion_particle_line import FusionParticle
from arp.detection_filter import LineFilter
from arp.line_extractor import LineExtractor

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)
g_fusion_filter = None
g_particle_filter = None

extractor = LineExtractor()

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='png',
        type=str
    )
    parser.add_argument(
        '--video',
        help='zmq or /path/to/video/file',
        default=None,
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


predict_time = []
process_time = []
show_img = None

#im is rgb
def hanle_frame(args, frameId, origin_im, im, logger, model, dataset, file_name):
    global predict_time, process_time, show_img
    logger.info('Processing frame: {}'.format(frameId))

    # cv2.imshow("tmplog", im)
    # cv2.waitKey(0)
    timers = defaultdict(Timer)
    t = time.time()
    im = im[:, :, ::-1]
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
            model, im, None, timers=timers
        )
    predict_time.append(time.time() - t)
    logger.info('Inference time: {:.3f}s'.format(time.time() - t))
    logger.info('predict_time: {:.3f}s'.format(np.mean(np.array(predict_time))))
    # for k, v in timers.items():
    #     logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
    if frameId == 1:
        logger.info(
            ' \ Note: inference on the first image will be slower than the '
            'rest (caches and auto-tuning need to warm up)'
        )

    t = time.time()
    img_debug = True
    ret = extractor.get_detection_line(
        im,
        cls_boxes,
        cls_segms,
        cls_keyps,
        dataset=dataset,
        show_class=True,
        thresh=0.8,
        kp_thresh=2,
        frame_id=frameId,
        img_debug = img_debug
    )
    im, mid_im, top_im, result, fork_pos = ret
    process_time.append(time.time() - t)
    logger.info('get_detection_line time: {:.3f}s'.format(time.time() - t))
    #
    logger.info('process_time: {:.3f}s'.format(np.mean(np.array(process_time))))
    line_list = None
    cache_list = None
    particles = None
    filter_list = None
    if not result is None:
        line_list, cache_list, filter_list, particles = add2MsgQueue(result, frameId, fork_pos, img_debug)
    g_debug_img_queue.put((origin_im[:, :, ::-1], im, mid_im, top_im, line_list, cache_list, filter_list, frameId, fork_pos, file_name))
    if g_debug_img_queue.full():
        try:
            g_debug_img_queue.get_nowait()
        except Empty:
            print ("Queue.Empty")


def drawParticles(image, particles):
    histogram = np.array([[i, 0] for i in range(500)])

    for index, p in enumerate(particles):
        if abs(p.x) > 100:
            continue
        meter_scale = (3.5/extractor.lane_wid)
        # histogram[index][0] = index + 100#int(p.x) / meter_scale
        histogram[int(p.x / meter_scale) + 150][1] += 1
    cv2.polylines(image, np.int32([np.vstack((histogram[:,0] + extractor.IMAGE_WID/2 - 150, histogram[:,1])).T]), False, (0, 0, 250), thickness=1)

def drawParabola(image, line_param, type, color):
    points = []
    for x in range(-800, 30, 10):
        points.append([line_param[0] * x**2 + line_param[1] * x + line_param[2], x])
    points = np.array(points)
    points[:,0] = points[:,0] + extractor.IMAGE_WID/2
    points[:,1] = points[:,1] + extractor.IMAGE_HEI
    points = cv2.perspectiveTransform(np.array([points], dtype='float32'), np.array(extractor.H_OP))

    offset_y = extractor.CUT_OFFSET_IMG[0]
    points = points[0]
    points[:,1] = points[:,1] + offset_y
    # print ("drawParabola points:" + str(points))

    parabola_im = np.zeros((extractor.IMAGE_HEI,extractor.IMAGE_WID,3), np.uint8)
    if type in ["yellow dashed", "yellow solid", "yellow solid solid", "yellow dashed dashed", "yellow dashed-solid", "yellow solid-dashed"]:
        cv2.polylines(parabola_im, np.int32([np.vstack((points[:,0], points[:,1])).T]), False, (0, 200, 200), thickness=2)
    elif type in ["boundary", "fork_edge", "handrail"]:
        cv2.polylines(parabola_im, np.int32([np.vstack((points[:, 0], points[:, 1])).T]), False, (0, 0, 200), thickness=4)
    else:
        cv2.polylines(parabola_im, np.int32([np.vstack((points[:,0], points[:,1])).T]), False, color, thickness=2)
    kernel = np.ones((5,5), np.float32) / 25
    parabola_im = cv2.filter2D(parabola_im, -1, kernel)
    # parabola_im = cv2.GaussianBlur(parabola_im, (16, 16),0)
    image = cv2.addWeighted(image, 1., parabola_im, 1., 0)
    return image

def add2MsgQueue(result, frameId, fork_x, img_debug):
    if (result is None) or len(result[0]) == 0:
        print ("error: len(line_list) == 0")
        return [], None

    full_line_list = []
    full_cache_list = []
    line_filter = [left_fork_filter]
    is_fork = (len(result) == 2)
    if is_fork:
        if not right_fork_filter.isAvialabel():
            right_fork_filter.reset(left_fork_filter.cache_list)
        line_filter.append(right_fork_filter)
    else:
        if right_fork_filter.isAvialabel():
            left_fork_filter.extend(right_fork_filter.cache_list)

    for index, parabola_param in enumerate(result):
        line_list = []
        for (line_param, line_type) in zip(parabola_param[0], parabola_param[1]):
            if abs(line_param[2]) > 500:
                print ("abs(line_param[2]) > 500")
                continue
            # line_info = {'curve_param':line_param[0:3].tolist(), 'type':line_type, 'score':line_param[3], 'x':line_param[4]}
            line_info = {'curve_param':line_param[0:3].tolist(), 'type':line_type, 'score':line_param[3], 'x':line_param[2], 'middle':line_param[5]}
            line_list.append(line_info)
        line_list, cache_list = line_filter[index].get_predict_list(line_list, frameId, fork_x[0] if is_fork else None, index==0)
        full_line_list.append(line_list)
        full_cache_list.append(cache_list)

    filter_list = None
    particles = None
    # filter_list = dr_filter(line_list)
    # ret = particle_filter(line_list)
    # if not ret is None:
    #     filter_list, particles = ret

    finalMessage = {'frame': frameId, 'timestamp': time.time(), 'is_fork': is_fork, 'line_list': full_line_list[0]}
    json_str = json.dumps(finalMessage)
    print ("finalMessage:", json_str)
    if g_detect_queue.full():
        g_detect_queue.get_nowait()
    g_detect_queue.put(json_str)
    return full_line_list, full_cache_list, filter_list, particles

def get_right_parabola(line_list):
    for index, line in enumerate(line_list):
        if line["curve_param"][2] > 0:
            ret = line["curve_param"][:]
            ret[2] = ret[2] % extractor.lane_wid
            return ret
    ret = line_list[-1]["curve_param"][:]
    ret[2] = ret[2] % extractor.lane_wid
    return ret

def particle_filter(line_list):
    global g_particle_filter
    if line_list is None or len(line_list) == 0:
        return None
    param = get_right_parabola(line_list)
    meter_scale = (3.5/extractor.lane_wid)
    x = param[2] * meter_scale
    if g_particle_filter is None or (time.time() - g_particle_filter.timestamp > 1):
        g_particle_filter = FusionParticle(x, g_dr_queue)
        g_particle_filter.start()
        return None
    t = time.time()
    # x_estimate, particles = g_particle_filter.update(x)
    x_estimate, particles = g_particle_filter.update(x, param)
    dalta_x = (x_estimate - x) / meter_scale
    print (str(time.time()-t) + "particle filter adjust x:" + str(dalta_x))
    filter_list = copy.deepcopy(line_list)
    for line in filter_list:
        line["curve_param"][2] += dalta_x
    return filter_list, particles

g_x_log = []
g_x_pred_log = []
g_x_est_log = []
g_x_time = []
def dr_filter(line_list):
    if line_list is None or len(line_list) == 0:
        return None
    global g_fusion_filter
    param = get_right_parabola(line_list)
    meter_scale = (3.5/extractor.lane_wid)
    x = param[2] * meter_scale
    avg_speed = []
    avg_angle = []
    for i in range(10):
        message = g_dr_queue.get(True)
        json_item = json.loads(message)
        avg_speed.append(json_item["speed"])
        avg_angle.append(json_item["steerAngle"])
    avg_speed = np.array(avg_speed)
    debug_angle = avg_angle[0]
    avg_angle = np.array(avg_angle)
    avg_speed = np.mean(avg_speed)
    avg_angle = np.mean(avg_angle)
    print ("g_dr_queue speed:{} angle:{}->{}".format(avg_speed, debug_angle, avg_angle))
    v = avg_speed
    wheel_theta = avg_angle / const.STEER_RATIO
    wheel_theta = math.radians(wheel_theta)
    car_theta = np.pi/2 + wheel_theta
    w = v/(const.WHEEL_BASE/np.sin(wheel_theta))
    # if car_theta < np.pi / 2:
    #     w = -w
    if g_fusion_filter is None or (time.time() - g_fusion_filter.timestamp > 1):
        g_fusion_filter = Fusion(x, v, w)
        print ("kalman filter recreate")
        return None

    t = time.time() - g_fusion_filter.timestamp
    # pre_estimate = g_fusion_filter.get_estimate()
    #x, v, w, t, parabola_param
    if len(g_x_time) == 0:
        g_x_time.append(t)
    else:
        g_x_time.append(t + g_x_time[-1])
    g_x_log.append(x)
    estimate_x = g_fusion_filter.update_step(x, v, w, t, param)
    predict_x = g_fusion_filter.get_predict()
    g_x_pred_log.append(predict_x)
    g_x_est_log.append(estimate_x)
    print("kalman filter: {} + {} --> {} ".format(x, predict_x, estimate_x))
    if len(g_x_log) % 100 == 0:
        np.savetxt('kalman_x.txt', g_x_log, newline=',', fmt=str("%s"))
        np.savetxt('kalman_x_pred.txt', np.array(g_x_pred_log), newline=',', fmt=str("%s"))
        np.savetxt('kalman_x_est.txt', np.array(g_x_est_log), newline=',', fmt=str("%s"))
        np.savetxt('kalman_x_time.txt', np.array(g_x_time), newline=',', fmt=str("%s"))
    dalta_x = (estimate_x - x) / meter_scale
    print ("kalman filter adjust x:" + str(dalta_x))
    filter_list = copy.deepcopy(line_list)
    for line in filter_list:
        line["curve_param"][2] += dalta_x
    return filter_list

left_fork_filter = LineFilter()
right_fork_filter = LineFilter()
def main(args):

    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    zmq_video = args.video == "zmq"
    frameId = 0
    print ("args.video:" + str(args.video))
    socket = None
    im_list = None
    ret = None
    if zmq_video:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:{}".format(const.PORT_IMAGE_OUT))
    elif os.path.isdir(args.video):
        im_list = glob.glob(args.video + '/*.' + args.image_ext)
        im_list.sort()
    else:
        # From virtual camera video and its associated timestamp file on Drive PX2,e.g."./lane/videofilepath.h264"
        cap = cv2.VideoCapture(args.video)
    im_file_index = frameId
    while True:
        file_name = ""
        if zmq_video:
            try:
                socket.send_string('req from detectron')
                print ("--------------------send!")
                message = socket.recv()
                print ("--------------------recv!" + str(time.time()))
                print("Received message length:" + str(len(message)) + " type:" + str(type(message)))
                if len(message) < 100:
                    continue
                img_np = np.fromstring(message, np.uint8)
                if const.CAMERA_TYPE != 2:
                    img_np = img_np.reshape((1208, 1920,3))
                else:
                    img_np = img_np.reshape((604, 960,3))

                print("nparr type:" + str(type(img_np)) + " shape:" + str(img_np.shape))
                ret = True
            except KeyboardInterrupt:
                print ("interrupt received, stopping...")
                socket.close()
                context.term()
                ret = False
                cap.release()
        elif os.path.isdir(args.video):
            if im_file_index >= len(im_list):
                break
            file_name = im_list[im_file_index].split("/")[-1].split(".")[0]
            img_np = cv2.imread(im_list[im_file_index])
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            im_file_index += 1
            ret = True
        else:
            ret, img_np = cap.read()

        frameId += 1
        # read completely or raise exception
        if not ret:
            print("cannot get frame")
            break
        if frameId < 0:
            continue
        if frameId % 1 == 0:
            t = time.time()
            print("time:" + str(t))
            time.sleep(0.001)
            #cv2.imwrite("tmp" + str(frameId) + ".png", img_np)
            if extractor.scale_size:
                # img_np = cv2.resize(img_np, dsize=img_np.shape/2, interpolation=cv2.INTER_CUBIC)
                img_np = img_np[::2]
                img_np = img_np[:,::2]
            origin_im = np.copy(img_np)

            img_np = img_np[extractor.CUT_OFFSET_IMG[0]:extractor.CUT_OFFSET_IMG[1], 0:extractor.IMAGE_WID]
            print ("detection size:", img_np.shape)
            # img_np = cv2.undistort(img_np, mtx, dist, None)
            hanle_frame(args, frameId, origin_im, img_np, logger, model, dummy_coco_dataset, file_name)
            logger.info('hanle_frame time: {:.3f}s'.format(time.time() - t))

    raw_input('press Enter to exit...')


def show_debug_img():
    print ("debug img process start !")
    while(True):
        message = g_debug_img_queue.get(True)
        if not message is None:
            origin_im, im, mid_im, top_im, line_list_array, cache_list_array, filter_list_array, frameId, fork_pos, file_name = message

            half_size = (int(im.shape[1] / 2), int(im.shape[0] / 2))
            if extractor.IMAGE_WID > 960:
                im = cv2.resize(im, half_size)
                top_im = cv2.resize(top_im, (extractor.IMAGE_WID/2, extractor.IMAGE_HEI/2))

                mid_im = cv2.resize(mid_im, half_size)
                # mid_im = mid_im[604:902, 0:extractor.IMAGE_WID]
                # mid_im = cv2.resize(mid_im, (int(extractor.IMAGE_WID / 2), 150))
            else:
                # mid_im = mid_im[302:451, 0:extractor.IMAGE_WID]
                pass

            if (not line_list_array is None) and (not cache_list_array is None):
                if filter_list_array is None:
                    filter_list_array = [[]] if len(line_list_array) == 1 else [[],[]]
                line_color = [(0, 200, 0), (100, 200, 0)]
                for line_list, cache_list, filter_list, color in zip(line_list_array, cache_list_array, filter_list_array, line_color):
                    x_pos = []
                    x_pos_11 = []
                    prob_wid = extractor.IMAGE_WID
                    if prob_wid > 960:
                        prob_wid = prob_wid / 2
                    for i in range(-int(prob_wid / 2), int(prob_wid / 2), 1):
                        matched_y = 1
                        matched_y_11 = 2
                        for l in line_list:
                            dis = abs(l['x'] - i)
                            if dis < 4:
                                # hei = dis
                                if l['type'] == "boundary":
                                    matched_y = int(220 * l['score'])
                                else:
                                    matched_y = int(190 * l['score'] - dis * dis)
                        for l in cache_list:
                            dis = abs(l['x'] - i)
                            if dis < 8:
                                matched_y_11 = int(200 * l['score'] - dis * dis)
                        x_pos.append([i + int(prob_wid / 2), matched_y])
                        x_pos_11.append([i + int(prob_wid / 2), matched_y_11])
                    # h = np.zeros((100, extractor.IMAGE_WID, 3))
                    cv2.polylines(origin_im, [np.array(x_pos)], False, (0, 255, 0))
                    cv2.polylines(origin_im, [np.array(x_pos_11)], False, (0, 0, 255))
                    # origin_im = np.flipud(origin_im)

                    # cv2.imshow('prob', h)
                    # cv2.waitKey(1)

                    for line in line_list:
                        line_param = line['curve_param']
                        line_type = line['type']
                        origin_im = drawParabola(origin_im, line_param[0:3], line_type, color=color)

                    if not filter_list is None:
                        for line in filter_list:
                            line_param = line['curve_param']
                            line_type = line['type']
                            origin_im = drawParabola(origin_im, line_param[0:3], line_type, color=(200, 0, 0))
                    overlay = origin_im.copy()
                    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
                    # for index in range(len(line_array)):
                    #     if index > 0:
                    #         left_line = line_array[index - 1]
                    #         right_line = line_array[index]
                    #         fill_points = np.array([np.append(left_line, right_line[::-1], axis=0)], dtype=np.int32)
                    #         print ("fill_points:" + str(fill_points.shape))
                    #         print ("color[index - 1]:" + str(color[index - 1]))
                    #         cv2.fillPoly(overlay, fill_points, color[index - 1])
                    # alpha = 0.2
                    # cv2.addWeighted(overlay, alpha, origin_im, 1-alpha, 0, origin_im)

            # origin_im
            origin_im = np.append(origin_im, top_im, axis=1)
            im = np.append(im, mid_im, axis=1)
            show_img = np.append(origin_im, im, axis=0)
            file_name = "source_{}_{}.png".format(file_name, frameId)
            cv2.imwrite(os.path.join(args.output_dir, file_name), show_img)
            cv2.imshow("carlab", show_img)
            cv2.waitKey(1)


def result_sender():
    print ("sender process start !")
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.SNDTIMEO, 3000)
    socket.bind("tcp://*:{}".format(const.PORT_DETECTION))
    while(True):
        message = g_detect_queue.get(True)
        if not message is None:
            recv = socket.recv()
            print ("Received request:%s" % recv)
            try:
                socket.send(message)
            except zmq.ZMQError:
                time.sleep(1)

def dr_recever():
    print ("dr recever process start !")
    sub_context = zmq.Context()
    socket = sub_context.socket(zmq.SUB)
    print ("tcp://localhost:{}".format(const.PORT_DR_OUT))
    socket.connect("tcp://localhost:{}".format(const.PORT_DR_OUT))
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    # socket.setsockopt(zmq.CONFLATE, 1)
    while(True):
        try:
            string = socket.recv()
            # print ("Received:{}".format(len(string)))
            if g_dr_queue.full():
                g_dr_queue.get(True)
            g_dr_queue.put(string)

        except zmq.ZMQError, Queue.em:
            time.sleep(1)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    # g_fusion_filter = Fusion()
    g_detect_queue = Queue(2)
    g_dr_queue = Queue(10)
    p = Process(target=result_sender)
    p.start()


    g_debug_img_queue = Queue(2)
    p = Process(target=show_debug_img)
    p.start()
    # pdr_receiver = Process(target=dr_recever)
    # pdr_receiver.start()
    main(args)

