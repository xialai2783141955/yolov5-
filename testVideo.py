# -*- coding: utf-8 -*-
import argparse
import random

import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, check_img_size, scale_coords
from utils.torch_utils import time_synchronized, select_device

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='./weights/personcar-best.pt',
                    help='model.pt path(s)')  # 模型路径仅支持.pt文件
parser.add_argument('--img-size', type=int, default=480, help='inference size (pixels)')  # 检测图像大小，仅支持480
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')  # 置信度阈值
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')  # NMS阈值
# 选中运行机器的GPU或者cpu，有GPU则GPU，没有则cpu，若想仅使用cpu，可以填cpu即可
parser.add_argument('--device', default='',
                    help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--save-dir', type=str, default='inference', help='directory to save results')  # 文件保存路径
parser.add_argument('--classes', nargs='+', type=int,
                    help='filter by class: --class 0, or --class 0 2 3')  # 分开类别
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')  # 使用NMS
opt = parser.parse_args()  # opt局部变量，重要
out, weight, imgsz = opt.save_dir, opt.weights, opt.img_size  # 得到文件保存路径，文件权重路径，图像尺寸
device = select_device(opt.device)  # 检验计算单元,gpu还是cpu
half = device.type != 'cpu'  # 如果使用gpu则进行半精度推理

model = attempt_load(weight, map_location=device)  # 读取模型
imgsz = check_img_size(imgsz, s=model.stride.max())  # 检查图像尺寸
if half:  # 如果是半精度推理
    model.half()  # 转换模型的格式
names = model.module.names if hasattr(model, 'module') else model.names  # 得到模型训练的类别名

colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]  # 给每个类别一个颜色
img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # 创建一个图像进行预推理
_ = model(img.half() if half else img) if device.type != 'cpu' else None  # 预推理


def predict(img):
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    t1 = time_synchronized()
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                               agnostic=opt.agnostic_nms)
    t2 = time_synchronized()
    InferNms = round((t2 - t1), 2)

    return pred, InferNms


def plot_one_box(img, x, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


if __name__ == '__main__':
    # video_path = 0
    video_path = "./UI_rec/test_/JapanStreet.mp4"
    # 初始化视频流
    vs = cv2.VideoCapture(video_path)
    (W, H) = (None, None)
    frameIndex = 0  # 视频帧数

    try:
        prop = cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        # print("[INFO] 视频总帧数：{}".format(total))
    # 若读取失败，报错退出
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ret, frame = vs.read()
    vw = frame.shape[1]
    vh = frame.shape[0]
    print("[INFO] 视频尺寸：{} * {}".format(vw, vh))
    output_video = cv2.VideoWriter("./results.avi", fourcc, 20.0, (vw, vh))  # 处理后的视频对象

    # 遍历视频帧进行检测
    while True:
        # 从视频文件中逐帧读取画面
        (grabbed, image) = vs.read()

        # 若grabbed为空，表示视频到达最后一帧，退出
        if not grabbed:
            print("[INFO] 运行结束...")
            output_video.release()
            vs.release()
            exit()
        # 获取画面长宽
        if W is None or H is None:
            (H, W) = image.shape[:2]

        image = cv2.resize(image, (850, 500))
        img0 = image.copy()
        img = letterbox(img0, new_shape=imgsz)[0]
        img = np.stack(img, 0)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        pred, useTime = predict(img)

        det = pred[0]
        p, s, im0 = None, '', img0
        if det is not None and len(det):  # 如果有检测信息则进入
            det[:, :4] = scale_coords(img.shape[1:], det[:, :4], im0.shape).round()  # 把图像缩放至im0的尺寸
            number_i = 0  # 类别预编号
            detInfo = []
            for *xyxy, conf, cls in reversed(det):  # 遍历检测信息
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                # 将检测信息添加到字典中
                detInfo.append([names[int(cls)], [c1[0], c1[1], c2[0], c2[1]], '%.2f' % conf])
                number_i += 1  # 编号数+1

                label = '%s %.2f' % (names[int(cls)], conf)

                # 画出检测到的目标物
                plot_one_box(image, xyxy, label=label, color=colors[int(cls)])

        # 实时显示检测画面
        cv2.imshow('Stream', image)
        image = cv2.resize(image, (vw, vh))
        output_video.write(image)  # 保存标记后的视频
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # print("FPS:{}".format(int(0.6/(end-start))))
        frameIndex += 1

        # if frameIndex >= total:  # 可设置检测的最大帧数提前退出
        #     print("[INFO] 运行结束...")
        #     output_video.release()
        #     vs.release()
        #     exit()
