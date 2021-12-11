# -*- coding: utf-8 -*-
# @TIME : 2021/7/27 16:10
# @AUTHOR : Xu Bai
# @FILE : dispatch_pros.py
# @DESCRIPTION :废弃的多进程处理
import argparse
import threading
from track import detect
from multiprocessing import Queue, Process, Pool


def get_pool(args):
    p = Pool(5)  # 设置进程池的大小
    p.map(detect, iterable=(args, args, args))
    p.close()  # 关闭进程池
    p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--source', type=str,
                        default='inference/input/mot_1min.mp4', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', default=False,
                        help='display results')
    parser.add_argument('--save-txt', action='store_true', default=False,
                        help='save results to *.txt')
    # class 0 is person default=[0]
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[1, 2, 3, 5, 7], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument("--dd", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    get_pool(args)
    print('ths process is ended')
