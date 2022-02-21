#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 10:13:59 2021

@author: vince
"""

import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmcv.parallel import MMDataParallel
from mmdet.apis import set_random_seed, train_detector, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmdet.apis import init_detector, inference_detector
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


# config_file = "/home/vince/repos/CBNetV2/config_finetune.py"
config_file = "/home/vince/repos/CBNetV2/config_finetune.py"
work_dir = '/media/vince/DataStorageHDD/Models/mmdetection/finetuning/cascade_mask_rcnn_cbv2_swinS_no_synth'
# work_dir = '/media/vince/DataStorageHDD/Models/mmdetection/finetuning/cascade_mask_rcnn_cbv2_swin_small/epoch_16'
# checkpoint = "/media/vince/DataStorageHDD/Models/mmdetection/mask_rcnn_cbv2_swinT/epoch_17.pth"
checkpoint = "/media/vince/DataStorageHDD/Models/mmdetection/cascade_mask_rcnn_cbv2_swin_small/epoch_16.pth"
finetuned_checkpoint = '/media/vince/DataStorageHDD/Models/mmdetection/finetuning/cascade_mask_rcnn_cbv2_swin_small/epoch_80.pth'
finetuned_checkpoint = '/media/vince/DataStorageHDD/Models/mmdetection/finetuning/cascade_mask_rcnn_cbv2_swinS_no_synth/fold_01/epoch_100.pth'
resume_from = True
seed = None
deterministic = False
no_validate = False
show_score_thr = 0.3   # (default: 0.3)

def train_fold(fold_number):
    cfg = Config.fromfile(config_file)
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.work_dir = work_dir
    
    cfg.load_from = checkpoint
    # cfg.resume_from = finetuned_checkpoint
    
    cfg.gpu_ids = range(1) 

    distributed = False
    
    cfg.checkpoint_config = dict(interval=20)
    
    cfg.runner['max_epochs'] = 100
    cfg.optimizer['lr'] = 0.00005
    cfg.lr_config = {'policy': 'step', 'warmup': 'linear', 'warmup_iters': 500, 'warmup_ratio': 0.001, 'step': [90]}
    cfg.data['samples_per_gpu'] = 2

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

    # create fold directory
    fold = "/fold_" + str(fold_number).rjust(2,'0')
    dir_fold = work_dir + fold
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(dir_fold))
    cfg.work_dir = dir_fold
    
    train_fold_set = "/train_fold_" + str(fold_number).rjust(2,'0')
    val_fold_set = "/val_fold_" + str(fold_number).rjust(2,'0')
    test_fold_set = "/test_fold_" + str(fold_number).rjust(2,'0')
    
    cfg.data.train.ann_file = '/home/vince/repos/coco-annotator/datasets/annotations' + fold + train_fold_set + '.json'
    cfg.data.val.ann_file = '/home/vince/repos/coco-annotator/datasets/annotations' + fold + val_fold_set + '.json'
    cfg.data.test.ann_file = '/home/vince/repos/coco-annotator/datasets/annotations' + fold + test_fold_set + '.json'
    
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()           

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
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
            
    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    
    # # # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # model.CLASSES = datasets[0].CLASSES
    
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    # wrap_fp16_model(model)
    ckpt = load_checkpoint(model, finetuned_checkpoint)
    
    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader, show=False, out_dir=None,  show_score_thr=show_score_thr)
    
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric="bbox"))
    metric = dataset.evaluate(outputs, **eval_kwargs)
    print(metric)
    metric_dict = dict(config=cfg, metric=metric)
    eval_kwargs.update(dict(metric="segm"))
    metric = dataset.evaluate(outputs, **eval_kwargs)
    print(metric)
    metric_dict = dict(config=cfg, metric=metric)


if __name__ == '__main__':
    for i in range(1, 5):
        train_fold(i)
    
    
    # EVALUATION
    # in case the test dataset is concatenated
    # samples_per_gpu = 1
    # if isinstance(cfg.data.test, dict):
    #     cfg.data.test.test_mode = True
    #     samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
            
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
    # checkpoint = load_checkpoint(model, finetuned_checkpoint)
    
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
    # device = 'cuda:0'
    # checkpoint_file = "/media/vince/DataStorageHDD/Models/mmdetection/mask_rcnn_cbv2_swin_tiny/epoch_1.pth"
    # config_model = 'configs/cbnet/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
    # # init a detector
    # model = init_detector(cfg, checkpoint_file, device=device)
    # # inference the demo image
    # output = inference_detector(model, '/home/vince/Pictures/annotated_synth_original.png')
    
    # # show the results
    # show_result_pyplot(model, 'demo/demo.jpg', output, score_thr=0.5)
    
    
# if __name__ == '__main__':
#     main()
