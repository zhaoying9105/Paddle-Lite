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

#include "operation/yolo_box.h"
#include <string>
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"
// #include "plugin_yolo_box.h"
#include <stdlib.h>
#include <dlfcn.h>
namespace nnadapter {
namespace cambricon_mlu {

int ConvertYoloBox(Converter* converter, core::Operation* operation) {
  YOLO_BOX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto imgsize_tensor = converter->GetMappedTensor(imgsize_operand);
  if (!imgsize_tensor) {
    imgsize_tensor = converter->ConvertOperand(imgsize_operand);
  }

if(converter->get_fusion_yolobox_multiclass_nms3_to_detection_output()){
  int image_height = 608;
  int image_width = 608;
  auto op_params = GetKeyValues(converter->op_params().c_str());
  if (op_params.count("image_height")) {
    image_height = stoi(op_params["image_height"]);
  }
  if (op_params.count("image_width")) {
    image_width = stoi(op_params["image_width"]);
  }
  auto img_shape_tensor = converter->AddInt32ConstantTensor(
      std::vector<int32_t>({image_height, image_width}).data(), {1, 2});
  auto yolo_box_node =
      converter->network()->AddIYoloBoxNode(input_tensor, img_shape_tensor);
  NNADAPTER_CHECK(yolo_box_node) << "Failed to add yolo_box node.";
  std::vector<float> anchors_fp32;
  for (int i = 0; i < anchors.size(); i++) {
    anchors_fp32.push_back(static_cast<float>(anchors[i]));
  }
  magicmind::Layout input_layout =
      ConvertToMagicMindDataLayout(input_operand->type.layout);
  yolo_box_node->SetLayout(input_layout);
  yolo_box_node->SetAnchorsVal(anchors_fp32);
  yolo_box_node->SetClassNumVal(static_cast<int64_t>(class_num));
  yolo_box_node->SetConfidenceThresholdVal(conf_thresh);
  yolo_box_node->SetDownsampleRatioVal(static_cast<int64_t>(downsample_ratio));
  yolo_box_node->SetClipBBoxVal(clip_bbox);
  yolo_box_node->SetScaleXYVal(scale_x_y);
  yolo_box_node->SetImageShape(image_height, image_width);
  auto boxes_tensor = yolo_box_node->GetOutput(0);
  auto scores_tensor = yolo_box_node->GetOutput(1);
  converter->UpdateTensorMap(boxes_operand, boxes_tensor);
  converter->UpdateTensorMap(scores_operand, scores_tensor);
}else{

  NNADAPTER_CHECK(false) << "mm plugin disabled(yolobox), this code should not be run";

  // // std::string lib_path = "/usr/local/neuware/lib64/libyolo_box_plugin.so";
  // // auto kernel_lib = dlopen(lib_path.c_str(),RTLD_LAZY);
  // // NNADAPTER_CHECK(kernel_lib) << "Failed to dlopen " << lib_path;
  // const std::string plugin_op_name = "PluginYoloBox";
  // magicmind::TensorMap yolo_box_input_map;
  // yolo_box_input_map["Input"] =
  //     std::vector<magicmind::ITensor*>({input_tensor});
  // yolo_box_input_map["ImgSize"] =
  //     std::vector<magicmind::ITensor*>({imgsize_tensor});

  // magicmind::DataTypeMap yolo_box_outputs_dtype;
  // yolo_box_outputs_dtype["Boxes"] = {magicmind::DataType::FLOAT32};
  // yolo_box_outputs_dtype["Scores"] = {magicmind::DataType::FLOAT32};

  // auto yolo_box_node = converter->network()->AddIPluginNode(
  //     plugin_op_name, yolo_box_input_map, yolo_box_outputs_dtype);
  // NNADAPTER_CHECK(yolo_box_node) << "Failed to add yolobox node.";
  // std::vector<int64_t> anchors_int64;
  // for (int i = 0; i < anchors.size(); i++) {
  //   anchors_int64.push_back(static_cast<int64_t>(anchors[i]));
  // }
  // yolo_box_node->SetAttr("class_num", (int64_t)class_num);
  // yolo_box_node->SetAttr("conf_thresh", conf_thresh);
  // yolo_box_node->SetAttr("downsample_ratio", (int64_t)downsample_ratio);
  // yolo_box_node->SetAttr("clip_bbox", clip_bbox);
  // yolo_box_node->SetAttr("scale_x_y", scale_x_y);
  // yolo_box_node->SetAttr("anchors", anchors_int64);
  // auto boxes_tensor = yolo_box_node->GetOutput(0);
  // auto scores_tensor = yolo_box_node->GetOutput(1);

  // auto perms_dim = magicmind::Dims({4});
  // int perms_value [4] = {0,1,3,2};
  // auto perms_node = converter->network()->AddIConstNode(magicmind::DataType::INT32,perms_dim ,perms_value);
  // NNADAPTER_CHECK(perms_node) << "Failed to add perms node for bboxes.";

  // int start_dim = 1;
  // int end_dim = 2;
  // auto flatten_start_node = converter->network()->AddIConstNode(magicmind::DataType::INT32,magicmind::Dims({1}),&start_dim);
  // NNADAPTER_CHECK(flatten_start_node) << "Failed to add flatten start node for bboxes";
  // auto flatten_end_node = converter->network()->AddIConstNode(magicmind::DataType::INT32,magicmind::Dims({1}),&end_dim);
  // NNADAPTER_CHECK(flatten_end_node) << "Failed to add flatten start node for bboxes";

  // // transpose and reshape for boxes_tensor
  // // from [batch, anchors_num,4, h_w] -transpose-> [batch, anchors_num,h_w, 4] -reshape-> [batch, anchors_num * h_w , 4]
  // auto bbox_permute_node = converter->network()->AddIPermuteNode(boxes_tensor,perms_node->GetOutput(0));
  // NNADAPTER_CHECK(bbox_permute_node) << "Failed to add permute node for bboxes";
  // auto  bbox_reshape_node = converter->network()->AddIFlattenNode(bbox_permute_node->GetOutput(0),flatten_start_node->GetOutput(0),flatten_end_node->GetOutput(0));
  // NNADAPTER_CHECK(bbox_reshape_node) << "Failed to add permute node for bboxes";

  // // transpose and reshape for scores_tensor
  // // from [batch, anchors_num,class_num, h_w] -transpose-> [batch, anchors_num,h_w, class_num] -reshape-> [batch, anchors_num * h_w , class_num]
  // auto scores_permute_node = converter->network()->AddIPermuteNode(scores_tensor,perms_node->GetOutput(0));
  // NNADAPTER_CHECK(scores_permute_node) << "Failed to add permute node for scores";
  // auto  scores_reshape_node = converter->network()->AddIFlattenNode(scores_permute_node->GetOutput(0),flatten_start_node->GetOutput(0),flatten_end_node->GetOutput(0));
  // NNADAPTER_CHECK(scores_reshape_node) << "Failed to add permute node for scores";

  // converter->UpdateTensorMap(boxes_operand, bbox_reshape_node->GetOutput(0));
  // converter->UpdateTensorMap(scores_operand, scores_reshape_node->GetOutput(0));
}
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
