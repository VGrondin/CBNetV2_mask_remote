#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 10:04:33 2021

@author: vince
"""

# The new config inherits a base config to highlight the necessary modification
# _base_ = 'configs/cbnet/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
_base_ = 'configs/cbnet/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=1),
#         mask_head=dict(num_classes=1))
#     )

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_head=dict(num_classes=1))
    )

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = '/media/vince/DataStorageHDD/Models/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth'
load_from = '/lustre06/project/6004732/spaceboy/models/pretrained/cascade_mask_rcnn/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth'

img_prefix = '/home/vince/repos/coco-annotator/datasets/essai_03/'


# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('tree',)
data = dict(
    train=dict(
        img_prefix='',
        classes=classes,
        ann_file='/home/vince/repos/detectron2_TreeGrasp/output/train_RGB_30vis_1000px_5kp_coco_02.json'),
    val=dict(
        img_prefix='',
        classes=classes,
        ann_file='/home/vince/repos/detectron2_TreeGrasp/output/val_RGB_30vis_1000px_5kp_coco_02.json'),
    test=dict(
        img_prefix='',
        classes=classes,
        ann_file='/home/vince/repos/detectron2_TreeGrasp/output/test_RGB_30vis_1000px_5kp_coco_02.json'))

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


# entire real set
# dataset_type = 'COCODataset'
# classes = ('Tree',)
# data = dict(
#     train=dict(
#         img_prefix='/home/vince/repos/coco-annotator/datasets/essai_03/',
#         classes=classes,
#         ann_file='/home/vince/repos/detectron2_TreeGrasp/output/essai_03-14.json'),
#     val=dict(
#         img_prefix='/home/vince/repos/coco-annotator/datasets/essai_03/',
#         classes=classes,
#         ann_file='/home/vince/repos/coco-annotator/datasets/annotations/fold_00/val_fold_00.json'),
#     test=dict(
#         img_prefix='/home/vince/repos/coco-annotator/datasets/essai_03/',
#         classes=classes,
#         ann_file='/home/vince/repos/coco-annotator/datasets/annotations/fold_00/test_fold_00.json'))




# model = dict(
#     roi_head=dict(
#         type='HybridTaskCascadeRoIHead',
#         interleaved=True,
#         mask_info_flow=True,
#         num_stages=3,
#         stage_loss_weights=[1, 0.5, 0.25],
#         bbox_roi_extractor=dict(
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[4, 8, 16, 32]),
#         bbox_head=[
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=1,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.1, 0.1, 0.2, 0.2]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
#                                loss_weight=1.0)),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=1,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.05, 0.05, 0.1, 0.1]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
#                                loss_weight=1.0)),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=1,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.033, 0.033, 0.067, 0.067]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
#         ],
#         mask_roi_extractor=dict(
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[4, 8, 16, 32]),
#         mask_head=[
#             dict(
#                 type='HTCMaskHead',
#                 with_conv_res=False,
#                 num_convs=4,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 num_classes=1,
#                 loss_mask=dict(
#                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
#             dict(
#                 type='HTCMaskHead',
#                 num_convs=4,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 num_classes=1,
#                 loss_mask=dict(
#                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
#             dict(
#                 type='HTCMaskHead',
#                 num_convs=4,
#                 in_channels=256,
#                 conv_out_channels=256,
#                 num_classes=1,
#                 loss_mask=dict(
#                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
#         ]))
