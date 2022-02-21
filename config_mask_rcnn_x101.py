# The new config inherits a base config to highlight the necessary modification
_base_ = [
    "configs/keypoints/mask_rcnn_x101_3x_coco.py", 	# single swin backbone
]

model = dict(
    backbone=dict(
	frozen_stages=1, # added
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1))
    )

load_from = '/lustre06/project/6004732/spaceboy/models/pretrained/mask_rcnn/mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco.pth'





