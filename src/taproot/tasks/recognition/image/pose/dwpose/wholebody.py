# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
import os
import numpy as np
import torch.nn as nn
import warnings

from typing import Union, Optional
from pathlib import Path

import copy

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.model.utils import revert_sync_batchnorm

from mmpose.apis import inference_topdown
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.structures import merge_data_samples
from mmpose.models.builder import build_pose_estimator

from mmdet.registry import MODELS
from mmdet.registry import DATASETS
from mmdet.apis import inference_detector

from taproot.util import inject_state_dict

# Copied from https://github.com/open-mmlab/mmpose/blob/main/mmpose/apis/inference.py
def init_pose_estimator(
    config: Union[str, Path, Config],
    checkpoint: Optional[str] = None,
    device: str = 'cuda:0',
    cfg_options: Optional[dict] = None
) -> nn.Module:
    """Initialize a pose estimator from a config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights. Defaults to ``None``
        device (str): The device where the anchors will be put on.
            Defaults to ``'cuda:0'``.
        cfg_options (dict, optional): Options to override some settings in
            the used config. Defaults to ``None``

    Returns:
        nn.Module: The constructed pose estimator.
    """

    if isinstance(config, (str, Path)):
        config = Config.fromfile(config, lazy_import=False)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None

    config.model.train_cfg = None

    # register all modules in mmpose into the registries
    scope = config.get('default_scope', 'mmpose')
    if scope is not None:
        init_default_scope(scope)

    model = build_pose_estimator(config.model)
    model = revert_sync_batchnorm(model)

    # get dataset_meta in this priority: checkpoint > config > default (COCO)
    dataset_meta = None

    if checkpoint is not None:
        inject_state_dict(checkpoint, model)

    if 'dataset_meta' in config.meta:
        # checkpoint from mmpose 1.x
        dataset_meta = config.meta['dataset_meta']

    if dataset_meta is None:
        dataset_meta = dataset_meta_from_config(config, dataset_mode='train')

    if dataset_meta is None:
        warnings.simplefilter('once')
        warnings.warn('Can not load dataset_meta from the checkpoint or the '
                      'model config. Use COCO metainfo by default.')
        dataset_meta = parse_pose_metainfo(
            dict(from_file='configs/_base_/datasets/coco.py'))

    model.dataset_meta = dataset_meta

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

# Copied from https://github.com/open-mmlab/mmdetection/blob/main/mmdet/apis/inference.py
def init_detector(
    config: Union[str, Path, Config],
    checkpoint: Optional[str] = None,
    palette: str = 'none',
    device: str = 'cuda:0',
    cfg_options: Optional[dict] = None,
) -> nn.Module:
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        palette (str): Color palette used for visualization. If palette
            is stored in checkpoint, use checkpoint's palette first, otherwise
            use externally passed palette. Currently, supports 'coco', 'voc',
            'citys' and 'random'. Defaults to none.
        device (str): The device where the anchors will be put on.
            Defaults to cuda:0.
        cfg_options (dict, optional): Options to override some settings in
            the used config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None

    scope = config.get('default_scope', 'mmdet')
    if scope is not None:
        init_default_scope(config.get('default_scope', 'mmdet'))

    model = MODELS.build(config.model)
    model = revert_sync_batchnorm(model)

    if checkpoint is None:
        warnings.simplefilter('once')
        warnings.warn('checkpoint is None, use COCO classes by default.')
        model.dataset_meta = {'classes': get_classes('coco')}
    else:
        inject_state_dict(checkpoint, model)

    checkpoint_meta = config.meta

    # save the dataset_meta in the model for convenience
    if 'dataset_meta' in checkpoint_meta:
        # mmdet 3.x, all keys should be lowercase
        model.dataset_meta = {
            k.lower(): v
            for k, v in checkpoint_meta['dataset_meta'].items()
        }
    elif 'CLASSES' in checkpoint_meta:
        # < mmdet 3.x
        classes = checkpoint_meta['CLASSES']
        model.dataset_meta = {'classes': classes}
    else:
        warnings.simplefilter('once')
        warnings.warn(
            'dataset_meta or class names are not saved in the '
            'checkpoint\'s meta data, use COCO classes by default.')
        model.dataset_meta = {'classes': get_classes('coco')}

    # Priority:  args.palette -> config -> checkpoint
    if palette != 'none':
        model.dataset_meta['palette'] = palette
    else:
        test_dataset_cfg = copy.deepcopy(config.test_dataloader.dataset)
        # lazy init. We only need the metainfo.
        test_dataset_cfg['lazy_init'] = True
        metainfo = DATASETS.build(test_dataset_cfg).metainfo
        cfg_palette = metainfo.get('palette', None)
        if cfg_palette is not None:
            model.dataset_meta['palette'] = cfg_palette
        else:
            if 'palette' not in model.dataset_meta:
                warnings.warn(
                    'palette does not exist, random is used by default. '
                    'You can also set the palette to customize.')
                model.dataset_meta['palette'] = 'random'

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class Wholebody:
    def __init__(
        self,
        det_config=None,
        det_ckpt=None,
        pose_config=None,
        pose_ckpt=None,
        device="cpu",
    ):
        if det_config is None:
            det_config = os.path.join(
                os.path.dirname(__file__), "yolox_config/yolox_l_8xb8-300e_coco.py"
            )

        if pose_config is None:
            pose_config = os.path.join(
                os.path.dirname(__file__), "dwpose_config/dwpose-l_384x288.py"
            )

        if det_ckpt is None:
            det_ckpt = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"

        if pose_ckpt is None:
            pose_ckpt = "https://huggingface.co/wanghaofan/dw-ll_ucoco_384/resolve/main/dw-ll_ucoco_384.pth"

        # build detector
        self.detector = init_detector(det_config, det_ckpt, device=device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

        # build pose estimator
        self.pose_estimator = init_pose_estimator(pose_config, pose_ckpt, device=device)

    def to(self, device, dtype=None):
        self.detector.to(device, dtype=dtype)
        self.pose_estimator.to(device, dtype=dtype)
        return self

    def __call__(self, oriImg, *args, **kwargs):
        # predict bbox
        det_result = inference_detector(self.detector, oriImg)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
        )
        bboxes = bboxes[
            np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.5)
        ]

        # set NMS threshold
        bboxes = bboxes[nms(bboxes, 0.7), :4]

        # predict keypoints
        if len(bboxes) == 0:
            pose_results = inference_topdown(self.pose_estimator, oriImg)
        else:
            pose_results = inference_topdown(self.pose_estimator, oriImg, bboxes)
        preds = merge_data_samples(pose_results)
        preds = preds.pred_instances

        # preds = pose_results[0].pred_instances
        keypoints = preds.get("transformed_keypoints", preds.keypoints)
        if "keypoint_scores" in preds:
            scores = preds.keypoint_scores
        else:
            scores = np.ones(keypoints.shape[:-1])

        if "keypoints_visible" in preds:
            visible = preds.keypoints_visible
        else:
            visible = np.ones(keypoints.shape[:-1])
        keypoints_info = np.concatenate(
            (keypoints, scores[..., None], visible[..., None]), axis=-1
        )
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3
        ).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores, visible = (
            keypoints_info[..., :2],
            keypoints_info[..., 2],
            keypoints_info[..., 3],
        )

        return keypoints, scores
