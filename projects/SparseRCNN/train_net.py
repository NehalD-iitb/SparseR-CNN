#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
SparseRCNN Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
from torch.nn.parallel import DistributedDataParallel

import os
import itertools
import time
from typing import Any, Dict, List, Set


import torch
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import AutogradProfiler, DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping

from sparsercnn import SparseRCNNDatasetMapper, add_sparsercnn_config
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter

import logging
import wandb
import wandb_detectron

class Trainer(DefaultTrainer):
#     """
#     Extension of the Trainer class adapted to SparseRCNN.
#     """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = SparseRCNNDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = SparseRCNNDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


class WandbTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        # Note: we're not calling super here, because we want to skip DefaultTrainer's init.
        # TODO: Fix? To fix this we need to get rid of DefaultTrainer entirely. We could still
        #   delegate to it's methods, by assinging them to our methods.
        SimpleTrainer.__init__(self, model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks

        # Use wandb checkpointer instead of fvcore checkpointer. This checkpointer
        # automatically makes an input artifact reference for the loaded model
        self.checkpointer = wandb_detectron.WandbCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def build_writers(self):
        # Add wandb writer to save training metrics
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            wandb_detectron.WandbWriter()
            # Don't use the tensorboard writer because it clears logged images, and so do we.
        ]

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(wandb_detectron.WandbModelSaveHook(
            self.cfg.OUTPUT_DIR,
            # TODO: get from config
            {'bbox.AP': wandb_detectron.UP_IS_BETTER},
            self.cfg.SOLVER.CHECKPOINT_PERIOD,
            self.cfg.TEST.EVAL_PERIOD,
            self.cfg.TEST.EVAL_PERIOD
        ))
        return hooks

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_sparsercnn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    register_coco_instances("minicoco_train", {}, "/root/data/miniCOCO/instances_train2021.json", "/root/data/miniCOCO/images")

    # register_coco_instances("dvfsurgery_train", {}, "/root/data/DuralAVF/annotations/instances_train2017.json", "/root/data/DuralAVF/train2017")

    MetadataCatalog.get("minicoco_train").thing_classes = ['vehicles','people', 'animals']

    DatasetCatalog.get("minicoco_train")    


    register_coco_instances("minicoco_val", {}, "/root/data/miniCOCO/instances_val2021.json", "/root/data/miniCOCO/images")

    MetadataCatalog.get("minicoco_val").thing_classes = ['vehicles','people', 'animals']

    DatasetCatalog.get("minicoco_val")  



    # register_coco_instances("dvfsurgery_train", {}, "/root/data/dvfsurgery_synthetic/instances_v2_train2021.json", "/root/data/dvfsurgery_synthetic/images")

    # # register_coco_instances("dvfsurgery_train", {}, "/root/data/DuralAVF/annotations/instances_train2017.json", "/root/data/DuralAVF/train2017")

    # MetadataCatalog.get("dvfsurgery_train").thing_classes = ['nonsegmented','artery', 'blood','spinalcord','bluntprobe','scissor']

    # DatasetCatalog.get("dvfsurgery_train")    


    # register_coco_instances("dvfsurgery_val", {}, "/root/data/DuralAVF/annotations/instances_test2017.json", "/root/data/DuralAVF/test2017")

    # MetadataCatalog.get("dvfsurgery_val").thing_classes = ['nonsegmented','artery', 'blood','spinalcord','bluntprobe','scissor']

    # DatasetCatalog.get("dvfsurgery_val")  

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res



    if args.eval_only:
        run = wandb.init(config=cfg, job_type='eval')
        model = WandbTrainer.build_model(cfg)
        wandb_detectron.WandbCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = WandbTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(WandbTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        for file in glob.glob(os.path.join(cfg.OUTPUT_DIR, 'inference', '*')):
            eval_artifact = wandb.Artifact(
                type='result',
                name='run-%s-%s' % (run.id, os.path.basename(file)))
            with eval_artifact.new_file('dataset.json') as f:
                # TODO: we should use the URI for whatever our input artifact
                #   ended up being, rather than what's passed in via test
                # TODO: This writes an array, because there can be more than one
                #   test dataset. How should we handle that case?
                # TODO: standardize how to do this.
                json.dump({'dataset_artifact': cfg.DATASETS.TEST}, f)
            eval_artifact.add_file(file)
            wandb.run.log_artifact(eval_artifact)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    # TODO: track --eval-only from args
    # TODO: only do this on main process?
    wandb.init(config=cfg, job_type='train')
    trainer = WandbTrainer(cfg)




    # # Comment the below when not required
    # distributed = comm.get_world_size() > 1
    # if distributed:
    #     model = DistributedDataParallel(
    #         model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
    #     )

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
