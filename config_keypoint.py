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
    'configs/_base_/datasets/coco_detection.py',
    'configs/_base_/schedules/schedule_1x.py', 'configs/_base_/default_runtime.py',
    "configs/keypoints/faster_rcnn_r50_fpn_keypoints.py",
    # "configs/keypoints/faster_rcnn_hrnetv2p_w32_1x.py"
]
# _base_ = ["configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py", "configs/keypoints/faster_rcnn_r50_fpn_keypoints.py" ]

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1))
        # mask_head=dict(num_classes=1))
    )

# Modify dataset related settings
# dataset_type = 'COCODataset'
# classes = ('tree',)
# data = dict(
#     train=dict(
#         img_prefix='',
#         classes=classes,
#         ann_file='/home/vince/repos/detectron2_TreeGrasp/output/train_RGB_30vis_1000px_5kp_coco_02.json'),
#     val=dict(
#         img_prefix='',
#         classes=classes,
#         ann_file='/home/vince/repos/detectron2_TreeGrasp/output/val_RGB_30vis_1000px_5kp_coco_02.json'),
#     test=dict(
#         img_prefix='',
#         classes=classes,
#         ann_file='/home/vince/repos/detectron2_TreeGrasp/output/test_RGB_30vis_1000px_5kp_coco_02.json'))

# dataset_type = 'COCODataset'
# classes = ('Tree',)
# data = dict(
#     train=dict(
#         img_prefix='/home/vince/repos/coco-annotator/datasets/essai_03/',
#         classes=classes,
#         ann_file='/home/vince/repos/coco-annotator/datasets/annotations/fold_00/train_fold_00.json'),
#     val=dict(
#         img_prefix='/home/vince/repos/coco-annotator/datasets/essai_03/',
#         classes=classes,
#         ann_file='/home/vince/repos/coco-annotator/datasets/annotations/fold_00/val_fold_00.json'),
#     test=dict(
#         img_prefix='/home/vince/repos/coco-annotator/datasets/essai_03/',
#         classes=classes,
#         ann_file='/home/vince/repos/coco-annotator/datasets/annotations/fold_00/test_fold_00.json'))





