#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 10:04:33 2021

@author: vince
"""

# The new config inherits a base config to highlight the necessary modification
# _base_ = 'configs/cbnet/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
# _base_ = 'configs/cbnet/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.py'
# _base_ = "configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
_base_ = [
    # 'configs/_base_/datasets/coco_detection.py',
    # 'configs/_base_/schedules/schedule_1x.py', 'configs/_base_/default_runtime.py',
    "configs/keypoints/mask_rcnn_cbv2_swinT_3x_coco.py", 	# dual swin backbone
    #"configs/keypoints/mask_rcnn_swinS_3x_coco.py", 	# single swin backbone
    # "configs/keypoints/faster_rcnn_hrnetv2p_w32_1x.py"
]
# _base_ = ["configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py", "configs/keypoints/faster_rcnn_r50_fpn_keypoints.py" ]

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    backbone=dict(
	frozen_stages=3, # added
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1))
    )

load_from = '/lustre06/project/6004732/spaceboy/models/pretrained/mask_rcnn/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth'
#load_from = '/lustre06/project/6004732/spaceboy/models/pretrained/mask_rcnn/mask_rcnn_swin_small_patch4_window7.pth'




