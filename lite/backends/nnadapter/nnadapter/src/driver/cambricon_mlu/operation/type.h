// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

namespace nnadapter {

enum {
  /**
   * Custom Multi-class non maximum suppression (NMS) on a batched of boxes and scores.
   * In the NMS step, this operator greedily selects a subset of detection
   * bounding boxes that have high scores larger than score_threshold,
   * if providing this threshold, then selects the largest nms_top_k confidences
   * scores if nms_top_k is larger than -1.
   * Then this operator pruns away boxes that have high IOU (intersection over
   * union) overlap with already selected boxes by adaptive threshold NMS based
   * on parameters of nms_threshold and nms_eta.
   * Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
   * per image if keep_top_k is larger than -1.
   * https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/layers/multiclass_nms_cn.html
   *
   * Inputs:
   * * 0: bboxes, a NNADAPTER_FLOAT32 tensor.
   * * Two types of bboxes are supported:
   *     1. A 3-D Tensor with shape [N, M, 4/8/16/24/32] represents
   * the predicted locations of M bounding bboxes, N is the batch size. 
   *        Each bounding box has four coordinate values and the layout
   * is [xmin, ymin, xmax, ymax], when box size equals to 4.
   *     2. A 3-D Tensor with shape [M, C, 4]. M is the number of bounding
   * boxes, C is the class number.
   * * 1: scores,scores, a NNADAPTER_FLOAT32 tensor.
   * * Two types of scores are supported:
   *     1. A 3-D Tensor with shape [N, C, M] represents the predicted
   * confidence predictions.
   *        N is the batch size, C is the class number, M is number of bounding
   * boxes.
   *        For each category there are total M scores which corresponding M
   * bounding boxes.
   *        In this case, input bboxes should be the first case with shape [N,
   * M, 4/8/16/24/32].
   *     2. A 2-D LoDTensor with shape [M, C]. M is the number of bbox, C is the
   * class number.
   *        In this case, input bboxes should be the second case with shape [M,
   * C, 4].
   * * 2: rois_num(optional), a NNADAPTER_INT32 tensor with shape [B], B is the
   * number of images.
   * rois_nums exist only if bboxes and scores is in the second case.
   * * 3: background_label, a NNADAPTER_INT32 tensor with shape [1], the index
   * of background label.
   * If set to 0, the background label will be ignored.
   * If set to -1, then all categories will be considered.
   * * 4: score_threshold, a NNADAPTER_FLOAT32 tensor with shape [1], threshold
   * to filter out bounding boxes with low confidence score.
   * * 5: nms_top_k, a NNADAPTER_INT32 tensor with shape [1], maximum number of
   * detections to be kept according to the confidences after the filtering
   * detections based on score_threshold.
   * * 6: nms_threshold, a NNADAPTER_FLOAT32 tensor with shape [1], the
   * parameter for NMS.
   * * 7: nms_eta, a NNADAPTER_FLOAT32 tensor with shape [1], the parameter for
   * adaptive NMS.
   * * 8: keep_top_k, a NNADAPTER_INT32 tensor with shape [1], number of total
   * bboxes to be kept per image after NMS step.
   * "-1" means keeping all bboxes after NMS step.
   * * 9: normalized, a NNADAPTER_BOOL8 tensor with shape [1], whether
   * detections are normalized.
   * * 10: return_index,  a NNADAPTER_BOOL8 tensor with shape [1], whether to
   * return index of RoIs.
   *
   * Outputs:
   * * 0: output, a tensor with the same type as bboxes and all scores, with shape [No, 6 + classnum].
   * "No" is the number of all RoIs. Each row has 6+classnum values: [label, confidence,
   * xmin, ymin, xmax, ymax, score0, score1, ...]
   * * 1: out_rois_num, a NNADAPTER_INT32 tensor of shape [B], B is the number
   * of images.
   * The number of NMS RoIs in each image.
   * * 2: index, a NNADAPTER_INT32 tensor with shape [No] represents the index
   * of selected bbox.
   * The index is the absolute index cross batches.
   * It is valid only if "return_index" is true.
   */
  NNADAPTER_NMS_WITH_ALL_SCORE = -100,
};  // Custom operations type

}  // namespace nnadapter
