# -*- coding: utf-8 -*-
# @TIME : 2021/7/27 20:39
# @AUTHOR : Xu Bai
# @FILE : getArgs.py
# @DESCRIPTION :
import argparse
import os
import sys
from multiprocessing import Queue, Process, Pool
from copy import deepcopy

sys.path.append('')
import torch
import json
from .track import detect
from yolov5_DeepSort.yolov5.utils.general import check_img_size
import time


#
# def getArgs(conf_path='schdule.json'):
#     # 这里是配置文件地址，也可以是来自网络的json
#     opts = json.load(open(conf_path))
#     return opts
def get_conf(conf_path='schdule_full.json'):
    opts = None
    with open(conf_path, encoding='utf-8') as f:
        opts = json.load(f)
    return opts


def start_detect_job(opts):
    print('start_detect_job\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='./yolov5_DeepSort/yolov5/weights/yolov5s.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='', help='source')
    parser.add_argument('--output', type=str, default='./yolov5_DeepSort/inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # class 0 is person default=[0]
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[1, 2, 3, 5, 7], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="./yolov5_DeepSort/deep_sort_pytorch/configs/deep_sort.yaml")

    args = parser.parse_known_args()

    args[0].img_size = check_img_size(args[0].img_size)
    sources = [
        './yolov5_DeepSort/inference/input/部分违停.mp4',
        './yolov5_DeepSort/inference/input/mot_1min.mp4',
        './yolov5_DeepSort/inference/input/部分违停2.mp4']

    fist_time = time.time()

    sources = opts['worker1']

    args1 = deepcopy(args[0])
    args2 = deepcopy(args[0])
    args3 = deepcopy(args[0])

    args1.__setattr__('source', sources['source1'])
    args2.__setattr__('source', sources['source2'])
    args3.__setattr__('source', sources['source3'])

    argsall = [args1, args2, args3]
    i = -1
    with torch.no_grad():
        print('detect.............................')
        # detect(args[0], s1='./yolov5_DeepSort/inference/input/部分违停.mp4',
        #        s2='./yolov5_DeepSort/inference/input/mot_1min.mp4'
        #        , s3='./yolov5_DeepSort/inference/input/部分违停.mp4', )
        detect(args[0], s1=r'rtmp://127.0.0.1/live',
               s2=r'rtmp://127.0.0.1/live'
               , s3=r'rtmp://127.0.0.1/live', )
    # import psutil
    # from subprocess import PIPE
    # while True:
    #     with torch.no_grad():
    #         i = (i + 1) % len(argsall)
    #         p = psutil.Popen(target=detect, args=(argsall[i],), )
    #         print('*******process {} started, args:{}...********'.format(str(p), str(argsall[i])))
    #         print(p.name())  # 获取进程名
    #         print(p.username())  # 获取用户名
    #         print(p.communicate())  # 获取进程运行内容
    #         print(p.cpu_times())  # 获取进程运行的cpu时间
    #     break

    # print(str(time.time() - fist_time))
    # if int(time.time() - fist_time) == 8:
    #     print()
    #     print()
    #     print()
    #     i = (i + 1) % len(argsall)
    #     p.kill()
    #     p.close()
    #     p.join()
    #     print('*****process {} terminated, args:{}...*****'.format(str(p), str(argsall[i])))
