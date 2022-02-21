#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 09:51:45 2022

@author: vince
"""

# The new config inherits a base config to highlight the necessary modification
_base_ = [
    "configs/keypoints/mask_rcnn_cbv2_swinT_3x_coco.py", 	# dual swin backbone
]

model = dict(
    backbone=dict(
	frozen_stages=3, # added
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1))
    )

load_from = '/lustre06/project/6004732/spaceboy/models/pretrained/mask_rcnn/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth'





