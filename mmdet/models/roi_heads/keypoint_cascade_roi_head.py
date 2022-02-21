import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any, List, Tuple, Union
from detectron2.layers import cat
from mmcv.runner import ModuleList

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .cascade_roi_head import CascadeRoIHead


_TOTAL_SKIPPED = 0

def _keypoints_to_heatmap(
        keypoints: torch.Tensor, rois: torch.Tensor, heatmap_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode keypoint locations into a target heatmap for use in SoftmaxWithLoss across space.
    
        Maps keypoints from the half-open interval [x1, x2) on continuous image coordinates to the
        closed interval [0, heatmap_size - 1] on discrete image coordinates. We use the
        continuous-discrete conversion from Heckbert 1990 ("What is the coordinate of a pixel?"):
        d = floor(c) and c = d + 0.5, where d is a discrete coordinate and c is a continuous coordinate.
    
        Arguments:
            keypoints: tensor of keypoint locations in of shape (N, K, 3).
            rois: Nx4 tensor of rois in xyxy format
            heatmap_size: integer side length of square heatmap.
    
        Returns:
            heatmaps: A tensor of shape (N, K) containing an integer spatial label
                in the range [0, heatmap_size**2 - 1] for each keypoint in the input.
            valid: A tensor of shape (N, K) containing whether each keypoint is in
                the roi or not.
        """
    
        if rois.numel() == 0:
            return rois.new().long(), rois.new().long()
        offset_x = rois[:, 0]
        offset_y = rois[:, 1]
        scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
        scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])
    
        offset_x = offset_x[:, None]
        offset_y = offset_y[:, None]
        scale_x = scale_x[:, None]
        scale_y = scale_y[:, None]
    
        x = keypoints[..., 0]
        y = keypoints[..., 1]
    
        x_boundary_inds = x == rois[:, 2][:, None]
        y_boundary_inds = y == rois[:, 3][:, None]
    
        x = (x - offset_x) * scale_x
        x = x.floor().long()
        y = (y - offset_y) * scale_y
        y = y.floor().long()
    
        x[x_boundary_inds] = heatmap_size - 1
        y[y_boundary_inds] = heatmap_size - 1
    
        valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
        vis = keypoints[..., 2] > 0
        valid = (valid_loc & vis).long()
    
        lin_ind = y * heatmap_size + x
        heatmaps = lin_ind * valid
    
        return heatmaps, valid
    
def heatmaps_to_keypoints(maps: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
    """
    Extract predicted keypoint locations from heatmaps.

    Args:
        maps (Tensor): (#ROIs, #keypoints, POOL_H, POOL_W). The predicted heatmap of logits for
            each ROI and each keypoint.
        rois (Tensor): (#ROIs, 4). The box of each ROI.

    Returns:
        Tensor of shape (#ROIs, #keypoints, 4) with the last dimension corresponding to
        (x, y, logit, score) for each keypoint.

    When converting discrete pixel indices in an NxN image to a continuous keypoint coordinate,
    we maintain consistency with :meth:`Keypoints.to_heatmap` by using the conversion from
    Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a continuous coordinate.
    """
    # The decorator use of torch.no_grad() was not supported by torchscript.
    # https://github.com/pytorch/pytorch/issues/44768
    maps = maps.detach()
    rois = rois.detach()

    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = (rois[:, 2] - rois[:, 0]).clamp(min=1)
    heights = (rois[:, 3] - rois[:, 1]).clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()

    num_rois, num_keypoints = maps.shape[:2]
    xy_preds = maps.new_zeros(rois.shape[0], num_keypoints, 4)

    width_corrections = widths / widths_ceil
    height_corrections = heights / heights_ceil

    keypoints_idx = torch.arange(num_keypoints, device=maps.device)

    for i in range(num_rois):
        outsize = (int(heights_ceil[i]), int(widths_ceil[i]))
        roi_map = F.interpolate(
            maps[[i]], size=outsize, mode="bicubic", align_corners=False
        ).squeeze(
            0
        )  # #keypoints x H x W

        # softmax over the spatial region
        max_score, _ = roi_map.view(num_keypoints, -1).max(1)
        max_score = max_score.view(num_keypoints, 1, 1)
        tmp_full_resolution = (roi_map - max_score).exp_()
        tmp_pool_resolution = (maps[i] - max_score).exp_()
        # Produce scores over the region H x W, but normalize with POOL_H x POOL_W,
        # so that the scores of objects of different absolute sizes will be more comparable
        roi_map_scores = tmp_full_resolution / tmp_pool_resolution.sum((1, 2), keepdim=True)

        w = roi_map.shape[2]
        pos = roi_map.view(num_keypoints, -1).argmax(1)

        x_int = pos % w
        y_int = (pos - x_int) // w

        assert (
            roi_map_scores[keypoints_idx, y_int, x_int]
            == roi_map_scores.view(num_keypoints, -1).max(1)[0]
        ).all()

        x = (x_int.float() + 0.5) * width_corrections[i]
        y = (y_int.float() + 0.5) * height_corrections[i]

        xy_preds[i, :, 0] = x + offset_x[i]
        xy_preds[i, :, 1] = y + offset_y[i]
        xy_preds[i, :, 2] = roi_map[keypoints_idx, y_int, x_int]
        xy_preds[i, :, 3] = roi_map_scores[keypoints_idx, y_int, x_int]

    return xy_preds

@HEADS.register_module()
class KeypointCascadeRoIHead(CascadeRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self, output_heatmaps=False, keypoint_decoder=None, num_keypoints=5, **kwargs):
        super().__init__(**kwargs)
        self.output_heatmaps = output_heatmaps
        if keypoint_decoder:
            self.keypoint_decoder = build_head(keypoint_decoder)
        else:
            assert output_heatmaps is True
            self.keypoint_decoder = None

    # def init_keypoint_head(self, keypoint_roi_extractor, keypoint_head):
        self.with_keypoint = True
        self.share_roi_extractor = False
        
        keypoint_roi_extractor = dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32])
        
        # self.keypoint_roi_extractor = build_roi_extractor(keypoint_roi_extractor)
        
        keypoint_head=dict(
            type='KeypointRCNNHead',
            num_convs=8,
            in_channels=256,
            features_size=[256, 256, 256, 256],
            conv_out_channels=512,
            num_keypoints=num_keypoints,
            loss_keypoint=dict(type='MSELoss', loss_weight=5.0))
        # self.keypoint_head = build_head(keypoint_head)
        
        self.init_keypoint_head(keypoint_roi_extractor, keypoint_head)
        
        
    def init_keypoint_head(self, keypoint_roi_extractor, keypoint_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.keypoint_head = nn.ModuleList()
        if not isinstance(keypoint_head, list):
            keypoint_head = [keypoint_head for _ in range(self.num_stages)]
        assert len(keypoint_head) == self.num_stages
        for head in keypoint_head:
            self.keypoint_head.append(build_head(head))
            
        # keypoint_roi_extractor = dict(
        #     type='SingleRoIExtractor',
        #     roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
        #     out_channels=256,
        #     featmap_strides=[4, 8, 16, 32])
        if keypoint_roi_extractor is not None:
            self.share_roi_extractor = False
            self.keypoint_roi_extractor = ModuleList()
            if not isinstance(keypoint_roi_extractor, list):
                keypoint_roi_extractor = [
                    keypoint_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(keypoint_roi_extractor) == self.num_stages
            for roi_extractor in keypoint_roi_extractor:
                self.keypoint_roi_extractor.append(
                    build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.keypoint_roi_extractor = self.bbox_roi_extractor

    def init_weights(self, pretrained):
        super().init_weights(pretrained)
        if self.with_keypoint and self.keypoint_head:
            self.keypoint_head.init_weights()

    def forward_dummy(self, x, proposals):
        outs = super().forward_dummy(x, proposals)
        # keypoints head
        if self.with_keypoint:
            pass

        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_keypoints=None,
                      gt_masks=None,
                      heatmaps=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposals (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[i]
                bbox_sampler = self.bbox_sampler[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    rcnn_train_cfg)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                mask_results = self._mask_forward_train(
                    i, x, sampling_results, gt_masks, rcnn_train_cfg,
                    bbox_results['bbox_feats'])
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{i}.{name}'] = (
                        value * lw if 'loss' in name else value)
                    
            # keypoint head forward and loss
            if self.with_keypoint:
                keypoint_results = self._keypoint_forward_train(
                    i, x, sampling_results, bbox_results['bbox_feats'], gt_keypoints,
                    heatmaps, img_metas, gt_bboxes)
                if keypoint_results['loss_keypoint'] is not None:
                    if i==0:
                        losses.update(s0_loss_keypoint=keypoint_results['loss_keypoint'].unsqueeze(0) * lw)
                    if i==1:
                        losses.update(s1_loss_keypoint=keypoint_results['loss_keypoint'].unsqueeze(0) * lw)
                    if i==2:
                        losses.update(s2_loss_keypoint=keypoint_results['loss_keypoint'].unsqueeze(0) * lw)
                    
                # if keypoint_results['loss_keypoint'] is not None:
                #     for name, value in keypoint_results['loss_keypoint'].unsqueeze(0).items():
                #         losses[f's{i}.{name}'] = (
                #             value * lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                # bbox_targets is a tuple
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score = self.bbox_head[i].loss_cls.get_activation(
                            cls_score)
                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[i].num_classes,
                        cls_score[:, :-1].argmax(1), roi_labels)
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)


        return losses

    def _keypoint_forward_train(self, stage, x, sampling_results, bbox_feats,
                                gt_keypoints, heatmaps, img_metas, gt_bboxes):
        pos_rois_all = []
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            pos_rois_all.append(pos_rois)
            # if pos_rois.shape[0] == 0:
            #     return dict(loss_keypoint=None)
            keypoint_results = self._keypoint_forward_2(stage, x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            if pos_inds.shape[0] == 0:
                return dict(loss_keypoint=None)
            keypoint_results = self._keypoint_forward_2(
                stage, x, pos_inds=pos_inds, bbox_feats=bbox_feats)
        
        
        # 
        num_gt_instances = []
        num_props = []
        heatmaps = []
        valid = []
        for im_in_batch, res in enumerate(sampling_results):
            num_gt_instances.append(len(gt_keypoints[im_in_batch]))
            num_props.append(res.pos_bboxes.shape[0])
            keypoints = gt_keypoints[im_in_batch]
            heatmaps_per_image, valid_per_image = _keypoints_to_heatmap(
                    keypoints.reshape(-1,3),
                    res.pos_bboxes,
                    # gt_bboxes[im_in_batch][instances_per_image].unsqueeze(0),
                    56
                    )
            # heatmaps_per_image : a tensor of shape (N, K) containing an integer spatial label
            # in the range [0, heatmap_size**2 - 1] for each keypoint in the input
            heatmaps.append(heatmaps_per_image.view(-1))
            # valid_per_image : a tensor of shape (N, K) containing whether 
            # each keypoint is in the roi or not.
            valid.append(valid_per_image.view(-1))
            
            # DEBUG
            # heatmaps_gt_56x56 = torch.zeros(1, 5, 56, 56)
            # # create heatmap using gt (might need to inverse / and mod)
            # heatmaps_gt_56x56[0, 0, int(heatmaps_per_image[0][0]/56), int(heatmaps_per_image[0][0]%56) ] = 1   # 56*X + Y =  heatmaps_per_image[0][0]
            # heatmaps_gt_56x56[0, 1, int(heatmaps_per_image[0][1]/56), int(heatmaps_per_image[0][1]%56) ] = 1   # 56*X + Y =  heatmaps_per_image[0][0]
            # heatmaps_gt_56x56[0, 2, int(heatmaps_per_image[0][2]/56), int(heatmaps_per_image[0][2]%56) ] = 1   # 56*X + Y =  heatmaps_per_image[0][0]
            # heatmaps_gt_56x56[0, 3, int(heatmaps_per_image[0][3]/56), int(heatmaps_per_image[0][3]%56) ] = 1   # 56*X + Y =  heatmaps_per_image[0][0]
            # heatmaps_gt_56x56[0, 4, int(heatmaps_per_image[0][4]/56), int(heatmaps_per_image[0][4]%56) ] = 1   # 56*X + Y =  heatmaps_per_image[0][0]
            # gt_from_heatmaps = heatmaps_to_keypoints(heatmaps_gt_56x56, gt_bboxes[im_in_batch][instances_per_image].cpu().clone().unsqueeze(0))
            # print(gt_from_heatmaps[0,:,:2])
            # print(gt_keypoints[im_in_batch][instances_per_image])
        
        
        if len(heatmaps):
            keypoint_targets = cat(heatmaps, dim=0)
            # heatmaps_gt = cat(heatmaps_gt, dim=1)
            valid_all = cat(valid, dim=0).to(dtype=torch.uint8)                
            valid = torch.nonzero(valid_all).squeeze(1)
        
        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if len(heatmaps) == 0 or valid.numel() == 0:
            global _TOTAL_SKIPPED
            _TOTAL_SKIPPED += 1
            keypoint_results.update(loss_keypoint=keypoint_results['heatmaps'].sum() * 0, keypoint_targets=gt_keypoints)
            return keypoint_results
    
        N, K, H, W = keypoint_results['heatmaps'].shape
        pred_keypoint_logits = keypoint_results['heatmaps'].view(N * K, H * W)
    
        valid_preds = []
        idx_prop = 0 # starts at 1 because 0modX would increment it anyways 
        idx_kp = 0 # starts at one for modulo
        idx_gt = 0
        idx_kp_tot = 0
        for _, val in enumerate(valid_all):
            if idx_gt < len(num_props) - 1:
                if idx_kp == (num_props[idx_gt] * num_gt_instances[idx_gt] * K):
                    idx_gt += 1
                    idx_kp = 0
                    # print(idx_prop)
                    # idx_prop -= 1   # modulo 0 will add 1
            # get 
            # next proposal 
            if idx_kp%(K*num_gt_instances[idx_gt]) == 0:
                idx_prop += 1
                
            if val > 0:
                valid_preds.append((idx_prop-1)*K + idx_kp%K)
                
            idx_kp += 1
            idx_kp_tot += 1
    
        if pred_keypoint_logits.shape[0] < ((idx_prop-1)*K + idx_kp_tot%K-1):
            print('out of bound from valid ' + str(pred_keypoint_logits.shape[0]) + ' < '  + str((idx_prop-1)*K + idx_kp_tot%K-1))
            print('Number of proposals = ' + str(pred_keypoint_logits.shape[0]) + ', idx_prop = ' + str((idx_prop-1)*K))
            print('Number of heatmaps = ' + str(len(valid_all)) + ', idx_kp = ' + str(idx_kp_tot))
            
        
        
        loss_keypoint = F.cross_entropy(
            pred_keypoint_logits[valid_preds], keypoint_targets[valid], reduction="sum"
        )
        # loss_keypoint = keypoint_results['heatmaps'].sum() * 0
    
        # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
        # if normalizer is None:
        normalizer = valid.numel()
        loss_keypoint /= normalizer
        
        # loss_keypoint = self.keypoint_head.loss(keypoint_results['heatmaps'],
        #                                         heatmap, 0)
        keypoint_results.update(
            loss_keypoint=loss_keypoint, keypoint_targets=gt_keypoints)
        return keypoint_results

    def _keypoint_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        keypoint_pred = self.keypoint_head(x)
        keypoint_results = dict(heatmaps=keypoint_pred)
        return keypoint_results
    
    def _keypoint_forward_2(self, stage, x, rois=None, pos_inds=None, bbox_feats=None):
        """Keypoint head forward function used in both training and testing."""
        keypoint_roi_extractor = self.keypoint_roi_extractor[stage]
        keypoint_head = self.keypoint_head[stage]
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            keypoints_feats = keypoint_roi_extractor(
                x[:keypoint_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                keypoints_feats = self.shared_head(keypoints_feats)
        else:
            assert bbox_feats is not None
            keypoints_feats = bbox_feats[pos_inds]

        keypoint_pred = keypoint_head(keypoints_feats)
        
        keypoint_results = dict(heatmaps=keypoint_pred)
        return keypoint_results

    def simple_test_keypoints(self,
                              x,
                              img_metas,
                              proposals=None,
                              rcnn_test_cfg=None,
                              rescale=False):
        """Test only keypoints without augmentation."""
        assert self.keypoint_decoder is not None
        
        scale_factor = img_metas[0]['scale_factor']
        proposals[:,1] = proposals[:,1] * scale_factor[0]
        proposals[:,2] = proposals[:,2] * scale_factor[1]
        proposals[:,3] = proposals[:,3] * scale_factor[0]
        proposals[:,4] = proposals[:,4] * scale_factor[1]
        
        keypoint_results = self._keypoint_forward_2(x, rois=proposals)

        # Convert heatmaps to keypoints
        pred_keypoint_logits = keypoint_results['heatmaps']
        
        pred_from_heatmaps = torch.zeros(pred_keypoint_logits.shape[0], pred_keypoint_logits.shape[1], 4)
        for i in range(pred_keypoint_logits.shape[0]):
            # create heatmap using gt (might need to inverse / and mod)
            prop_boxes = torch.zeros(1,4)
            prop_boxes[0] = proposals[i,1:] #* 0.3125
            pred_from_heatmaps[i, :] = heatmaps_to_keypoints(pred_keypoint_logits[i].unsqueeze(0), proposals[i,1:].unsqueeze(0))
            
            # Upscale keypoints to the original size
            pred_from_heatmaps[i, :, 0] /= scale_factor[0]
            pred_from_heatmaps[i, :, 1] /= scale_factor[1]
            
            # print(pred_from_heatmaps[i,:,:2])
        
        # pred = heatmaps_to_keypoints(pred_keypoint_logits, proposals[:,1:])
        
        # pred = self.keypoint_decoder(res)
        keypoint_results['keypoints'] = pred_from_heatmaps.cpu().numpy()
        # Upscale keypoints to the original size
        # pred[:, :, 0] /= scale_factor[0]
        # pred[:, :, 1] /= scale_factor[1]
        if self.output_heatmaps:
            keypoint_results['heatmaps'] = keypoint_results['heatmaps'].cpu(
            ).numpy()
        else:
            keypoint_results.pop('heatmaps')
        return keypoint_results

    async def async_test_keypoints(self,
                                   x,
                                   img_metas,
                                   proposals=None,
                                   rcnn_test_cfg=None,
                                   rescale=False):
        """Test only keypoints without augmentation."""
        assert self.keypoint_decoder is not None
        keypoint_results = self._keypoint_forward(x)
        scale_factor = img_metas[0]['scale_factor']

        # Convert heatmaps to keypoints
        res = keypoint_results['heatmaps']
        pred = self.keypoint_decoder(res)
        keypoint_results['keypoints'] = pred.cpu().numpy()
        # Upscale keypoints to the original size
        pred[:, :, 0] /= scale_factor[0]
        pred[:, :, 1] /= scale_factor[1]
        if self.output_heatmaps:
            keypoint_results['heatmaps'] = keypoint_results['heatmaps'].cpu(
            ).numpy()
        else:
            keypoint_results.pop('heatmaps')
        return keypoint_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        if self.with_bbox:
            det_bboxes, det_labels = await self.async_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
            bbox_results = bbox2result(det_bboxes, det_labels,
                                       self.bbox_head.num_classes)
        else:
            bbox_results = np.zeros((1, 0, 5))

        if not self.with_mask:
            segm_results = None
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))

        result = {'bbox': bbox_results, 'mask': segm_results}
        if self.with_keypoint:
            if self.keypoint_decoder is not None:
                kpts_results = self.async_test_keypoints(
                    x, img_metas, rescale=rescale)
                result.update(kpts_results)
        else:
            kpts_results = None

        return result

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_keypoint_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                if self.bbox_head[i].custom_activation:
                    cls_score = [
                        self.bbox_head[i].loss_cls.get_activation(s)
                        for s in cls_score
                    ]
                bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
                rois = torch.cat([
                    self.bbox_head[i].regress_by_class(rois[j], bbox_label[j],
                                                       bbox_pred[j],
                                                       img_metas[j])
                    for j in range(num_imgs)
                ])

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        if torch.onnx.is_in_onnx_export():
            return det_bboxes, det_labels
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
                mask_rois = bbox2roi(_bboxes)
                num_mask_rois_per_img = tuple(
                    _bbox.size(0) for _bbox in _bboxes)
                aug_masks = []
                for i in range(self.num_stages):
                    mask_results = self._mask_forward(i, x, mask_rois)
                    mask_pred = mask_results['mask_pred']
                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
                    aug_masks.append(
                        [m.sigmoid().cpu().numpy() for m in mask_pred])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_masks = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_masks, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results
            
        if self.with_keypoint:
            num_keypoints = 5
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                keypoint_results = [[[] for _ in range(num_keypoints)]
                                for _ in range(num_imgs)]
            else:
                # Already done in mask section
                if not self.with_keypoint:
                    if rescale and not isinstance(scale_factors[0], float):
                        scale_factors = [
                            torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                            for scale_factor in scale_factors
                        ]
                    _bboxes = [
                        det_bboxes[i][:, :4] *
                        scale_factors[i] if rescale else det_bboxes[i][:, :4]
                        for i in range(len(det_bboxes))
                    ]
                keypoint_rois = bbox2roi(_bboxes)
                num_keypoints_rois_per_img = tuple(
                    _bbox.size(0) for _bbox in _bboxes)
                aug_keypoints = []
                for i in range(self.num_stages):
                    keypoint_results = self._keypoint_forward_2(i, x, keypoint_rois)
                    keypoints_pred = keypoint_results['heatmaps']
                    # split batch mask prediction back to each image
                    keypoints_pred = keypoints_pred.split(num_keypoints_rois_per_img, 0)
                    aug_keypoints.append(
                        [m.sigmoid().cpu().numpy() for m in keypoints_pred])
                    
                # apply keypoint post-processing to each image individually
                num_keypoints = keypoint_results['heatmaps'].shape[1]
                keypoint_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        keypoint_results.append(
                            [[]
                             for _ in range(num_keypoints)])
                    else:
                        aug_keypoint = [mask[i] for mask in aug_keypoints]
                        merged_keypoints = merge_aug_masks(
                            aug_keypoint, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        
                        pred_from_heatmaps = torch.zeros(merged_keypoints.shape[0], merged_keypoints.shape[1], 4, device='cuda')
                        for i in range(merged_keypoints.shape[0]):
                            # create heatmap using gt (might need to inverse / and mod)
                            prop_boxes = torch.zeros(1,4)
                            prop_boxes[0] = _bboxes[0][i,:] 
                            pred_from_heatmaps[i, :] = heatmaps_to_keypoints(torch.Tensor(merged_keypoints[i]).unsqueeze(0).cuda(), _bboxes[0][i,:].unsqueeze(0))
                            
                            # Upscale keypoints to the original size
                            pred_from_heatmaps[i, :, 0] /= scale_factors[0][0]
                            pred_from_heatmaps[i, :, 1] /= scale_factors[0][1]
                        
                        keypoint_results.append(pred_from_heatmaps.cpu().numpy())
            ms_keypoint_result['ensemble'] = keypoint_results
                    

        if self.with_mask and not self.with_keypoint:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        elif self.with_mask and self.with_keypoint:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble'], ms_keypoint_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results

    def simple_test_old(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        # assert self.with_bbox, 'Bbox head must be implemented.'

        if self.with_bbox:
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
            bbox_results = bbox2result(det_bboxes, det_labels,
                                       self.bbox_head.num_classes)
        else:
            bbox_results = np.zeros((1, 0, 5))

        if self.with_mask:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
        else:
            segm_results = None

        result = {'bbox': bbox_results, 'mask': segm_results}
        if self.with_keypoint:
            if self.with_bbox:
                kpts_results = self.simple_test_keypoints(
                    x, img_metas, bbox2roi(det_bboxes), rescale=rescale)
            # need to rescale keypoints
            
            
            # else:
            #     kpts_results = self.simple_test_keypoints(x, img_metas,
            #         rescale=rescale)
            # if self.keypoint_decoder is not None:
            #     kpts_results = self.simple_test_keypoints(
            #         x, img_metas, rescale=rescale)
            result.update(kpts_results)
        else:
            kpts_results = None

        return result

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
        
    

