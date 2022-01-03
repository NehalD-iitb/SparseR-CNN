#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.modeling.roi_heads import build_roi_heads

from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss

from .loss import SetCriterion, HungarianMatcher
from .head import DynamicHead, DynamicConv
from  detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import matplotlib.pyplot as plt


__all__ = ["SparseRCNN"]


@META_ARCH_REGISTRY.register()
class SparseRCNN(nn.Module):
    """
    Implement SparseRCNN
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        self.num_proposals = cfg.MODEL.SparseRCNN.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        self.num_heads = cfg.MODEL.SparseRCNN.NUM_HEADS
        self.pooler_res = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.mask_on = cfg.MODEL.MASK_ON

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        
        # Build Proposals.
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)
        # # Nehal: Build mask proposals
        self.init_mask_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)

        
        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())

        # # Nehal: Build mask head
        # if self.mask_on: 
        #     self.mask_pooler = ROIPooler(
        #             output_size=self.pooler_res,
        #             scales=tuple(1.0 / self.backbone.output_shape()[k].stride for k in self.in_features),
        #             sampling_ratio=cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO,
        #             pooler_type=cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE,
        #         )
        #     self.mask_head = build_mask_head(cfg = cfg, input_shape = ShapeSpec(channels=256,
        #                     width=cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION,
        #                     height=cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION) )

        # mask

        # Loss parameters:
        class_weight = cfg.MODEL.SparseRCNN.CLASS_WEIGHT
        giou_weight = cfg.MODEL.SparseRCNN.GIOU_WEIGHT
        l1_weight = cfg.MODEL.SparseRCNN.L1_WEIGHT
        mask_weight = cfg.MODEL.SparseRCNN.MASK_WEIGHT
        no_object_weight = cfg.MODEL.SparseRCNN.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.SparseRCNN.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL

        # Build Criterion.
        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=class_weight, 
                                   cost_bbox=l1_weight, 
                                   cost_giou=giou_weight,
                                   use_focal=self.use_focal)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight, "loss_mask": mask_weight }
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        if self.mask_on:
            losses = ["labels", "boxes" , "masks"]

        else:
            losses = ["labels", "boxes"]
        
        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=no_object_weight,
                                      losses=losses,
                                      use_focal=self.use_focal)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)


        # if self.mask_on:
        #     # freeze gradients for class and box head
        #     for param in self.head.parameters():
        #         param.requires_grad = False
        #     for param in self.init_proposal_features.parameters():
        #         param.requires_grad = False
        #     for param in self.init_proposal_boxes.parameters():
        #         param.requires_grad = False



    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        img = images.tensor[0].permute(1,2,0).cpu().numpy()
        img = (img - np.min(img))/(np.max(img) - np.min(img))
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_fimagesrom_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()        
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        proposal_boxes = self.init_proposal_boxes.weight.clone() # num_proposals X 4
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :] # Proposal box is now initialzied to be size of largest image

        # Prediction.
        # outputs_class, outputs_coord = self.head(features, proposal_boxes, self.init_proposal_features.weight)
        outputs_class, outputs_coord, outputs_mask = self.head(features, proposal_boxes, self.init_proposal_features.weight, self.init_mask_proposal_features.weight) # ROI FEATURES ARE NOW (numhead X numproposals X 7 X 7)
        # roi_features = roi_features[0].view(-1,self.pooler_res,self.pooler_res,256).permute(0,3,1,2) # code for masks works only for 1 iterative head as of now
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_masks': outputs_mask[-1]}

        # if self.mask_on: 
        #     #mask branch
        #     # create list format for boxes
        #     pred_boxes = list()
        #     N = output['pred_boxes'].shape[0]
        #     for b in range(N):
        #         pred_boxes.append(Boxes(output['pred_boxes'][b]))

        #     mask_roi_features = self.mask_pooler(features, pred_boxes)

            # # self_att.
            # pro_features = self.init_mask_proposal_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
            # pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
            # pro_features = pro_features + self.dropout1(pro_features2)
            # pro_features = self.norm1(pro_features)

            #  = self.mask_inst_head(, mask_roi_features)

            #nn.Softmax(dim = 2)(pred_masks_)
            # output['pred_masks'] = pred_masks_ #nn.Softmax(dim = 2)(pred_masks_)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)



            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c}
                                         for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_mask[:-1])]

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            if self.mask_on:
                mask_pred = output["pred_masks"]
            else:
                mask_pred = [None]

#################SANITY CHECK STARTS #######
            # gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            # targets = self.prepare_targets(gt_instances)
#################SANITY CHECK ENDS #######

            results = self.inference(box_cls, box_pred, mask_pred, images.image_sizes)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            target['masks'] = targets_per_image.gt_masks.to(self.device)
            new_targets.append(target)

        return new_targets

    def inference(self, box_cls, box_pred, mask_pred, image_sizes, targets=None):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            mask_pred (Tensor): tensors of shape (batch_size, num_proposals, K, 28, 28)
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal:
            scores = torch.sigmoid(box_cls) # box_cls : [1 X numproposals X classes]
            labels = torch.arange(self.num_classes, device=self.device).unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1) # num_proposals X class : Tensor[0,1,2,0,1,2,.....]

            for i, (scores_per_image, box_pred_per_image, mask_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, mask_pred, image_sizes
            )):
                # scores_per_image : [num_proposals X class]
                # box_pred_per_image : [num_proposals X 4]
                # mask_pred_per_image :  [num_proposals X cls X 14 X 14]
                result = Instances(image_size)
                # scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False) # ORIG  This is extracting top K, which involves each proposal to belong to multiple categories
                ## shouldnt each proposal be responsible to one category 
                
                scores_per_image, indexes = torch.max(scores_per_image, dim =1 , keepdim = True) #NEW 
                # labels_per_image = labels[topk_indices] #ORIGINAL
                labels_per_image = indexes[:,0] #NEW
                # box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4) #ORIG
                # box_pred_per_image = box_pred_per_image[topk_indices] # ORIG

                result.pred_boxes = Boxes(box_pred_per_image)
                result.pred_classes = labels_per_image
                # print(labels_per_image)
                result.scores = scores_per_image[:,0] # NEW
                # result.scores = scores_per_image #ORIG
                # pred_masks_ = []
                # for it,cls in enumerate(labels_per_image):
                #     pred_masks_.append(mask_pred_per_image[it,cls,:,:])
                # predicted_masks = torch.stack(pred_masks_)

############## SANITY CHECK FOR POST PROCESSING

                # device = labels_per_image.get_device()
                # mask_side_len = 14
                # gt_masks = []
                # for t in targets:

                #     target_masks_resized = t['masks'].crop_and_resize(
                #         t['boxes_xyxy'], mask_side_len
                #     )
                #     gt_masks.append(target_masks_resized)
                
                # gt_masks = torch.cat(gt_masks)

                # gt_masks_tensor = torch.zeros(len(labels_per_image), 14, 14).to(device)
                # for ind,label in enumerate(targets[0]['labels']):
                #     indices_labels = labels_per_image == label 
                #     gt_masks_tensor[indices_labels] = gt_masks[ind].float()
                    

                # result.pred_masks = gt_masks_tensor

###################################
                # result.pred_boxes = Boxes(targets[0]['boxes_xyxy'].to(device))
                # result.pred_masks = gt_masks.float().to(device)
                # result.scores = torch.zeros(gt_masks.shape[0])
                # result.pred_classes = targets[0]['labels']
#################################
############## SANITY CHECK ENDS
                if self.mask_on:
                    # mask_pred_per_image = mask_pred_per_image.repeat(1, self.num_classes, 1,1).view(-1, 14,14) # ORIG
                    # mask_pred_per_image = mask_pred_per_image.view(-1, 28,28)
    
                    # result.pred_masks = mask_pred_per_image[topk_indices] # ORIG
                    result.pred_masks = mask_pred_per_image # NEW


                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                scores, labels, box_pred, image_sizes
            )):
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
