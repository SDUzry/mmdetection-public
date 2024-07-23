# Copyright (c) OpenMMLab. All rights reserved.
import copy
from mmengine.config import ConfigDict

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector

import torch
from torch import Tensor

@MODELS.register_module()
class MaskRCNN(TwoStageDetector):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self,
                 backbone: ConfigDict,
                 fc_head: ConfigDict,
                 rpn_head: ConfigDict,
                 roi_head: ConfigDict,
                 train_cfg: ConfigDict,
                 test_cfg: ConfigDict,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        self.fc_head = MODELS.build(fc_head)

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        x = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        roi_outs = self.roi_head.forward(x, rpn_results_list,
                                         batch_data_samples)
        results = results + (roi_outs, )
        return results

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        losses = dict()

        x = self.extract_feat(batch_inputs)

        fc_loss = self.fc_head(x, batch_data_samples)
        if fc_loss is not None:
            losses.update(fc_loss)

        selected_batch_data_samples = []
        selected_x = []
        for i in range(len(batch_data_samples)):
            if batch_data_samples[i].metainfo['is_detected_image']:
                selected_batch_data_samples.append(batch_data_samples[i])
                selected_x.append(i)
        selected_x = tuple(x[i].index_select(0, torch.tensor(selected_x).to(x[0].device)) for i in range(len(x)))

        if len(selected_batch_data_samples) == 0:
            return losses

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(selected_batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                selected_x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert selected_batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in selected_batch_data_samples
            ]

        roi_losses = self.roi_head.loss(selected_x, rpn_results_list,
                                        selected_batch_data_samples)
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        x = self.extract_feat(batch_inputs)

        classify_pre,classify_scores = self.fc_head.predict(x)
        # 如果数据样本中有class_id字段，则为val_step;反之为predict_step
        if batch_data_samples[0].get('class_id',False):
            for data_sample, pre in zip(batch_data_samples, classify_pre):
                data_sample.pre_class_ids = pre

            if batch_data_samples[0].get('proposals', None) is None:
                rpn_results_list = self.rpn_head.predict(x, batch_data_samples, rescale=False)
            else:
                rpn_results_list = [
                    data_sample.proposals for data_sample in batch_data_samples
                ]

            results_list = self.roi_head.predict(x, rpn_results_list, batch_data_samples, rescale=rescale)

            batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)

            return batch_data_samples
        # batch>1
        elif len(batch_data_samples)>1:
            
            for data_sample, class_pre, score_pre in zip(batch_data_samples, classify_pre, classify_scores):
                data_sample.pre_class_ids = class_pre
                data_sample.pre_class_scores = score_pre
            base_classes = list(self.dataset_meta['base_classes'])
            detection_classes = list(self.dataset_meta['detection_classes'])
            segmentation_classes = list(self.dataset_meta['segmentation_classes'])
            only_classify = True
            # 如果该批次数据仅需进行分类，则直接返回结果不进行预测bbox与mask
            for data_sample in batch_data_samples:
                if base_classes[data_sample.pre_class_ids[0]] in detection_classes or base_classes[data_sample.pre_class_ids[0]] in segmentation_classes:
                    only_classify = False
                    break
            if only_classify:
                return batch_data_samples
            else:
                if batch_data_samples[0].get('proposals', None) is None:
                    rpn_results_list = self.rpn_head.predict(x, batch_data_samples, rescale=False)
                else:
                    rpn_results_list = [data_sample.proposals for data_sample in batch_data_samples]

                results_list = self.roi_head.predict(x, rpn_results_list, batch_data_samples, rescale=rescale)

                batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)

                return batch_data_samples
        # batch为1时,进行剪枝
        else:
            pre_list = classify_pre[0]
            pre_scores = classify_scores[0]
            batch_data_samples[0].pre_class_ids = pre_list
            batch_data_samples[0].pre_class_scores = pre_scores
            pre = pre_list[0]
            # 非检测分割分支直接返回以减少推理时间
            base_classes = list(self.dataset_meta['base_classes'])
            detection_classes = list(self.dataset_meta['detection_classes'])
            segmentation_classes = list(self.dataset_meta['segmentation_classes'])
            if base_classes[pre] not in detection_classes and base_classes[pre] not in segmentation_classes:
                return batch_data_samples
            else:
                if batch_data_samples[0].get('proposals', None) is None:
                    rpn_results_list = self.rpn_head.predict(x, batch_data_samples, rescale=False)
                else:
                    rpn_results_list = [
                        data_sample.proposals for data_sample in batch_data_samples
                    ]

                results_list = self.roi_head.predict(x, rpn_results_list, batch_data_samples, rescale=rescale)

                batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)

                return batch_data_samples