#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 09:09:23 2021

@author: vince
"""
import cv2

from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = "/home/vince/repos/CBNetV2/config_custom.py"
checkpoint_file = '/media/vince/DataStorageHDD/Models/mmdetection/finetuning/cascade_mask_rcnn_cbv2_swin_small/epoch_80.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# video_path = '/home/vince/Videos/Forest_walk_YT.mp4'    # 30 fps native
video_path = '/media/vince/DataStorageHDD/Videos/dashcam_jour_00.MP4'

# Get one video frame 
# vcap = cv2.VideoCapture('/home/vince/Videos/Forest_walk_YT.mp4')
vcap = cv2.VideoCapture(video_path)

# get vcap property 
w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vcap.get(cv2.CAP_PROP_FPS))
n_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

# VIDEO
# Grab the stats from image1 to use for the resultant video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')   
video_writer = cv2.VideoWriter("pred_03.mp4",fourcc, 10, (w, h))

# test a video and show the results
video = mmcv.VideoReader(video_path)

nframes = 0
for frame in video:
    
    # 10 fps
    if nframes % 3 == 0:
        result = inference_detector(model, frame)
        detection_frame = model.show_result(frame, result, wait_time=1, score_thr=0.7)
        
        cv2.imshow('frame', detection_frame)
        video_writer.write(detection_frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    nframes += 1
    
video_writer.release()
vcap.release()
cv2.destroyAllWindows()