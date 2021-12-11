# !/usr/bin/env python
# encoding: utf-8
import sys

sys.path.insert(0, './yolov5_DeepSort/yolov5/')
sys.path.insert(0, './yolov5_DeepSort/yolov5/models')
from collections import defaultdict, deque
from yolov5_DeepSort.event_processing import upload, add_event
from yolov5_DeepSort.yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5_DeepSort.yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5_DeepSort.yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5_DeepSort.deep_sort_pytorch.utils.parser import get_config
from yolov5_DeepSort.deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

true_labels = {0: 'person',
               1: 'bicycle',
               2: 'car',
               3: 'motorcycle',
               4: 'airplane',
               5: 'bus',
               # 6:'train',
               7: 'truck',
               999: 'obj2cls error',
               9999: 'id2cls error'}

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

max_hold_time = 5  # 最大停留5s
hold_dict = defaultdict(lambda: time.time())
id2cls = defaultdict(lambda: 9999)
obj2cls = defaultdict(lambda: 999)
processed_id = deque(maxlen=50)  # 已经被处理后的id，最大长度可以修改
# 逆时针顺序
part_area = [[[700, 190], [720, 880], [1900, 860], [1900, 170], ]]
part_area = np.array(part_area)


def compute_iou(gt_box, b_box):
    '''
    计算iou
    :param gt_box: ground truth gt_box = [x0,y0,x1,y1]（x0,y0)为左上角的坐标（x1,y1）为右下角的坐标
    :param b_box: bounding box b_box 表示形式同上
    :return:
    '''
    width0 = gt_box[2] - gt_box[0]
    height0 = gt_box[3] - gt_box[1]
    width1 = b_box[2] - b_box[0]
    height1 = b_box[3] - b_box[1]
    max_x = max(gt_box[2], b_box[2])
    min_x = min(gt_box[0], b_box[0])
    width = width0 + width1 - (max_x - min_x)
    max_y = max(gt_box[3], b_box[3])
    min_y = min(gt_box[1], b_box[1])
    height = height0 + height1 - (max_y - min_y)
    interArea = width * height
    boxAArea = width0 * height0
    boxBArea = width1 * height1
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xywh2xyxy(x):
    # print('x', x)  # tensor([1160.50000,  575.00000,  275.00000,  158.00000])
    # x = x.unsqueeze(0)
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # print(y)
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    # print('y', y)
    return y


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        if time.time() - hold_dict[id] >= max_hold_time:
            color = (0, 0, 255)  # BGR
        cls = id2cls[id]
        cls_label = true_labels[int(float(cls))]
        label = '{}{:d}|{}'.format("", id, cls_label.upper())
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        current_time = time.strftime('%Y-%m%d-%H-%M-%S')
        if time.time() - hold_dict[id] >= max_hold_time and id not in processed_id:  # 超时且没有被处理的id
            # print('将图片证据保存到本地！')
            # TODO
            # 同一个id应该只记录一次
            img_path = 'pics/{}.jpg'.format(current_time)
            cv2.imwrite(img_path, img)
            # TODO
            # 这里添加处理代码
            upload_img_result = upload(img_path)
            if upload_img_result['code'] == 'SUCCESS_200':
                remote_img_path = upload_img_result['data']
                add_event_result = add_event(eventPoint='地下室监控_6号通道外', images=remote_img_path)
                if add_event_result['code'] == 'SUCCESS_200':
                    print('上报事件成功')
            processed_id.append(id)  # 将本id标记为已处理
            del hold_dict[id]  # 处理完成后重新计算本id的出现时间
    return img


def detect(opt, save_img=False, s1='', s2='', s3=''):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    webcam = True
    stime = time.time()
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    # attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    '''
    model = attempt_load(weights, map_location=device)[
        'model'].float()  # load to FP32'''

    model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    start_time = time.time()
    if webcam:
        print('loadStreams....')
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference

        dataset = LoadStreams(source, img_size=imgsz, s1=s1, s2=s2, s3=s3, start_time=start_time)
    else:
        # view_img = False#True
        # save_img = False#True
        print('LoadImages ----------')

        dataset = LoadImages(path=source, img_size=imgsz, s1=s1, s2=s2, s3=s3, start_time=start_time)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'
    stime = time.time()
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):

        # print(img.shape)  # (3, 384, 640)
        # print(im0s.shape)  # (1080, 1920, 3)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    # 根据检测到的每一个框的特征值作为键，cls作为值
                    objkey = x_c + y_c + bbox_w + bbox_h
                    # 根据检测的目标框存储cls 这个cls是tensor 要转str
                    obj2cls[objkey] = str(cls.item())
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)
                xywhs_ndarray = xywhs.numpy()
                for i in np.array(outputs):  # 这个是追踪
                    for j in xywhs_ndarray:  # 检测
                        cost = compute_iou(i[:4], xywh2xyxy(torch.Tensor(j)))
                        # print(i, j, cost)
                        if cost >= 0.8:
                            # print('这俩是同一个物体，将根据检测结果对ID和类别进行绑定')
                            x_c, y_c, bbox_w, bbox_h = j
                            objkey = x_c + y_c + bbox_w + bbox_h
                            id2cls[i[-1]] = obj2cls[objkey]

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    # 根据追踪到的box重新根据检测结果获得类别
                    # 此时应该再对所有id计时
                    for id_ in identities:
                        current_hold_time = time.time() - hold_dict[id_]
                        # print(hold_dict.items())
                        if current_hold_time >= max_hold_time:
                            print('\n【{}|{}】停留时间大于{}s，将根据策略做进一步处理！' \
                                  .format(id_, true_labels[int(float(id2cls[id_]))], max_hold_time))
                            # del hold_dict[id_]
                    draw_boxes(im0, bbox_xyxy, identities)

                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                # print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    # print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    # 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck'
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/yolov5s.pt', help='model.pt path')
    # file/folder, 0 for webcam
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
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)
