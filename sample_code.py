#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:29:51 2021

@author: vince
"""

from mmdet.apis import init_detector, inference_detector
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

# config_file = 'configs/cbnet/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
# config_file = 'configs/cbnet/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.py'
config_file = 'configs/cbnet/htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '/media/vince/DataStorageHDD/Models/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth'
# checkpoint_file = '/media/vince/DataStorageHDD/Models/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth'
checkpoint_file = '/media/vince/DataStorageHDD/Models/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth'

device = 'cuda:0'
checkpoint_file = "/media/vince/DataStorageHDD/Models/mmdetection/mask_rcnn_cbv2_swinT/epoch_17.pth"
config_file = "/home/vince/repos/CBNetV2/config_custom.py"
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)

im = '/home/vince/repos/coco-annotator/datasets/essai_03/image_02112_RGB.png'
im = '/home/vince/repos/coco-annotator/datasets/essai_03/image_02376_RGB.png'
# im = '/home/vince/Pictures/annotated_synth_original.png'
# inference the demo image
output = inference_detector(model, im)

# show the results
show_result_pyplot(model, im, output, score_thr=0.5)