#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 10:13:59 2021

@author: vince
"""

import copy
import os
import os.path as osp
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmcv.parallel import MMDataParallel
from mmdet.apis import set_random_seed, train_detector, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset
                            )
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmdet.apis import init_detector, inference_detector
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


# freeze backbone layer in 
# /home/vince/repos/CBNetV2/configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py

if __name__ == '__main__':
# def main():
    # config_file = "configs/keypoints/faster_rcnn_r50_fpn_keypoints.py"
    config_file =  "config_mask_keypoint.py"
    # work_dir = "/media/vince/DataStorageHDD/Models/mmdetection/mask_rcnn_cbv2_swinT"
    work_dir = '/media/vince/DataStorageHDD/Models/mmdetection/keypoint_models/mask_rcnn_keypoints'
    checkpoint = "/media/vince/DataStorageHDD/Models/mmdetection/mask_rcnn_cbv2_swinT/epoch_17.pth"
    # resume_from = True
    seed = None
    deterministic = False
    no_validate = True
    show_score_thr = 0.01   # (default: 0.3)


    cfg = Config.fromfile(config_file)
    
    # dataset_type = 'WflwDataset'
    data_root = '/home/vince/repos/coco-annotator/datasets/'    
    dataset_type = 'CocoDataset'
    classes = ('tree',)

    img_scale = (400, 400)
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    train_pipeline = [
        dict(type='LoadImageFromFile', to_float32=True),
        # dict(type='FocusOnFace', out_size=384),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_keypoint=True),
        dict(type='Resize', img_scale=img_scale, keep_ratio=False),
        # dict(type='PhotoMetricDistortion'),
        # dict(type='Corrupt', corruption='gaussian_noise'),
        # dict(type='Corrupt', corruption='jpeg_compression'),
        dict(type='RandomFlip', flip_ratio=0.),
        dict(type='Normalize', **img_norm_cfg),
        # dict(type='Pad', size_divisor=32),
        dict(type='KeypointsToHeatmaps', sigma=3),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_keypoints', 'heatmaps' ]), # 'gt_keypoints', 'heatmaps' 
    ]
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        # dict(type='FocusOnFace', out_size=384),
        dict(
            type='MultiScaleFlipAug',
            img_scale=img_scale,
            flip=False,
            transforms=[
                dict(type='Resize', img_scale=img_scale, keep_ratio=False),
                dict(type='RandomFlip', flip_ratio=0.),
                dict(type='Normalize', **img_norm_cfg),
                # dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]
    cfg.data = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        train=dict(
            type=dataset_type,
            classes=classes,
            ann_file='/home/vince/repos/detectron2_TreeGrasp/output/train_RGB_30vis_1000px_5kp_coco_02.json',  #'/home/vince/repos/detectron2_TreeGrasp/output/essai_03-14.json',
            img_prefix="", # data_root + 'essai_03/',
            pipeline=train_pipeline),
        val=dict(
            type=dataset_type,
            classes=classes,
            ann_file='/home/vince/repos/detectron2_TreeGrasp/output/val_RGB_30vis_1000px_5kp_coco_02.json',
            # img_prefix=data_root + 'essai_03/',
            pipeline=test_pipeline),
        test=dict(
            type=dataset_type,
            classes=classes,
            ann_file='/home/vince/repos/detectron2_TreeGrasp/output/test_RGB_30vis_1000px_5kp_coco_02.json',
            # img_prefix=data_root + 'essai_03/',
            pipeline=test_pipeline))
    evaluation = dict(interval=1, metric='keypoint', kpts_thr=0.1)
    
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.work_dir = work_dir
    
    cfg.load_from = checkpoint
    # cfg.resume_from = checkpoint
    
    cfg.gpu_ids = range(1) 

    distributed = False
    
    cfg.checkpoint_config = dict(interval=1)
    
    cfg.runner['max_epochs'] = 1
    cfg.optimizer['lr'] = 0.0001
    cfg.lr_config['step'] = [3, 9]
    

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    # cfg.dump(osp.join(cfg.work_dir, osp.basename(cfg)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as                     
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(config_file)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    # model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # train_detector(
    #     model,
    #     datasets,
    #     cfg,
    #     distributed=distributed,
    #     validate=no_validate,
    #     timestamp=timestamp,
    #     meta=meta)


    # EVALUATION
    # in case the test dataset is concatenated
    # samples_per_gpu = 1
    # if isinstance(cfg.data.test, dict):
    #     cfg.data.test.test_mode = True
    #     samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    #     if samples_per_gpu > 1:
    #         # Replace 'ImageToTensor' to 'DefaultFormatBundle'
    #         cfg.data.test.pipeline = replace_ImageToTensor(
    #             cfg.data.test.pipeline)
            
    # # build the dataloader
    # dataset = build_dataset(cfg.data.test)
    # data_loader = build_dataloader(
    #     dataset,
    #     samples_per_gpu=samples_per_gpu,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     dist=distributed,
    #     shuffle=False)
    

    # # # build the model and load checkpoint
    # cfg.model.train_cfg = None
    # model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # # model.CLASSES = datasets[0].CLASSES
    
    # # fp16_cfg = cfg.get('fp16', None)
    # # if fp16_cfg is not None:
    # # wrap_fp16_model(model)
    # checkpoint = load_checkpoint(model, checkpoint)
    
    # model = MMDataParallel(model, device_ids=[0])
    # outputs = single_gpu_test(model, data_loader, show=False, out_dir=None,  show_score_thr=show_score_thr)
    
    # eval_kwargs = cfg.get('evaluation', {}).copy()
    # # hard-code way to remove EvalHook args
    # for key in [
    #         'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
    #         'rule'
    # ]:
    #     eval_kwargs.pop(key, None)
    # eval_kwargs.update(dict(metric="bbox"))
    # metric = dataset.evaluate(outputs, **eval_kwargs)
    # print(metric)
    # metric_dict = dict(config=cfg, metric=metric)
    # eval_kwargs.update(dict(metric="segm"))
    # metric = dataset.evaluate(outputs, **eval_kwargs)
    # print(metric)
    # metric_dict = dict(config=cfg, metric=metric)
    
    
    # inference
    device = 'cuda:0'
    checkpoint_file = "/media/vince/DataStorageHDD/Models/mmdetection/keypoint_models/mask_rcnn_keypoints/epoch_1.pth"
    # init a detector
    model = init_detector(cfg, checkpoint_file, device=device)
    # inference the demo image
    im_name = '/media/vince/DataStorageHDD/Datasets/pseudo_label_maskRCNN/init_dashcam/dashcam_jour_00_1158.jpg'
    im_name = '/home/vince/Pictures/annotated_synth_original.png'
    im_name = '/home/vince/Pictures/tree_pred2.png'
    im_name = '/home/vince/Pictures/00002_img.png'
    
    output = inference_detector(model, im_name)
    
    # show the results
    show_result_pyplot(model, im_name, (output['bbox'][0], output['mask'][0]), score_thr=0.7)
    
    pred_img = model.show_result(im_name, (output['bbox'][0], output['mask'][0]), score_thr=0.7, show=False)
    
    # add keypoint to visualization
    for i in range(len(output["keypoints"])):
        for j in range(len(output["keypoints"][0])):
            pred_img = cv.circle(pred_img, center=(int(output["keypoints"][i][j][0]), int(output["keypoints"][i][j][1])), radius=4, color=(255,0,0), thickness=-1)
         
    
    cv.imshow('pred_image', pred_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
# if __name__ == '__main__':
#     main()
    
    