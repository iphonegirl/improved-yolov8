#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zhangyanwei  
# @Creat Time : 2023/8/7 11:46
# @File: train.py  
# @Project: yolov8-dev

from ultralytics import YOLO
import ultralytics
import argparse

ultralytics.checks()


def run():
    parser = argparse.ArgumentParser('YOLOv8 Training Script')
    parser.add_argument('--cfg', type=str, default='./costom/config/coco-training-config.yaml',
                        help='train config file path')
    parser.add_argument('--model', type=str, default='./costom/models/yolov8s.yaml',
                        help='pretrained weights file path')
    parser.add_argument('--project', type=str, default='')
    args = parser.parse_args()

    print(args.project,args.cfg,args.model)
    model = YOLO(args.model)
    model.train(cfg=args.cfg, project=args.project)

if __name__ == '__main__':
    run()
