// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "driver/cambricon_mlu/engine.h"
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <limits>
#include <utility>
#include <vector>
#include <dlfcn.h>
#include "driver/cambricon_mlu/converter.h"
#include "driver/cambricon_mlu/optimizer/convert_datalayout_nchw_to_nhwc.h"
#include "driver/cambricon_mlu/optimizer/fix_non_max_suppression.h"
#include "driver/cambricon_mlu/optimizer/fix_quantized_ops.h"
#include "driver/cambricon_mlu/optimizer/fix_transpose.h"
#include "optimizer/constant_fold_operations.h"
#include "optimizer/fuse_matmul_add_into_fully_connected.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace cambricon_mlu {

Context::Context(void* device, const char* properties) : device_(device) {
  // Extract the build parameters from the context properties
  NNADAPTER_LOG(INFO) << "properties: " << std::string(properties);
  std::string build_config_file_path;
  auto key_values = GetKeyValues(properties);
  if (key_values.count(CAMBRICON_MLU_BUILD_CONFIG_FILE_PATH)) {
    build_config_file_path = key_values[CAMBRICON_MLU_BUILD_CONFIG_FILE_PATH];
  } else {
    build_config_file_path =
        GetStringFromEnv(CAMBRICON_MLU_BUILD_CONFIG_FILE_PATH);
  }
  if (!build_config_file_path.empty()) {
    build_config_file_path_ = build_config_file_path;
    NNADAPTER_LOG(INFO) << "BuildConfig file path: " << build_config_file_path_;
  } else {
    NNADAPTER_LOG(INFO) << "Build model with default config.";
  }

  std::string op_params_file_path;
  if (key_values.count(CAMBRICON_MLU_OP_PARAMS_FILE_PATH)) {
    op_params_file_path = key_values[CAMBRICON_MLU_OP_PARAMS_FILE_PATH];
  } else {
    op_params_file_path = GetStringFromEnv(CAMBRICON_MLU_OP_PARAMS_FILE_PATH);
  }
  if (!op_params_file_path.empty()) {
    op_params_file_path_ = op_params_file_path;
    NNADAPTER_LOG(INFO) << "op_params file path: " << op_params_file_path_;
  } else {
    NNADAPTER_LOG(INFO) << "op convert with default value.";
  }
  if (key_values.count(
          CAMBRICON_MLU_FUSION_YOLOBOX_MULTICLASS_NMS3_TO_DETECTION_OUTPUT)) {
    std::string tmp_str = key_values
        [CAMBRICON_MLU_FUSION_YOLOBOX_MULTICLASS_NMS3_TO_DETECTION_OUTPUT];
    if (strcmp(tmp_str.c_str(), "false") == 0 ||
        strcmp(tmp_str.c_str(), "0") == 0) {
      fusion_yolobox_multiclass_nms3_to_detection_output = false;
    } else {
      fusion_yolobox_multiclass_nms3_to_detection_output = true;
    }
  } else {
    fusion_yolobox_multiclass_nms3_to_detection_output =
        GetBoolFromEnv(CAMBRICON_MLU_FUSION_YOLOBOX_MULTICLASS_NMS3_TO_DETECTION_OUTPUT);
  }
}

Context::~Context() {}

Program::~Program() {
  if(enable_mm_profile_){
    profiler_->Stop();
    NNADAPTER_LOG(WARNING) << "magicmind profile stop...";
    profiler_->Destroy();
    NNADAPTER_LOG(WARNING) << "magicmind profile destory...";
  }
  Clear();
  if (queue_) {
    MLU_CNRT_CHECK(cnrtQueueDestroy(queue_));
    queue_ = nullptr;
  }
}

void Program::Clear() {
  tensors_.clear();
  input_types_.clear();
  output_types_.clear();
  input_names_.clear();
  inputs_perm_.clear();
  model_buffer_.clear();
  dump_graph_path_ = "";
  dump_graph_buffer_ = nullptr;
}

int Program::Build(core::Model* model, core::Cache* cache) {
  // load mm plugins
  std::string yolo_box_lib_path = "/usr/local/neuware/lib64/libyolo_box_plugin.so";
  auto yolo_box_kernel_lib = dlopen(yolo_box_lib_path.c_str(),RTLD_LAZY);
  NNADAPTER_CHECK(yolo_box_kernel_lib) << "Failed to dlopen " << yolo_box_lib_path;

  std::string nms_lib_path = "/usr/local/neuware/lib64/libmulticlass_nms3_plugin.so";
  auto nms_kernel_lib = dlopen(nms_lib_path.c_str(),RTLD_LAZY);
  NNADAPTER_CHECK(nms_kernel_lib) << "Failed to dlopen " << nms_lib_path;

  std::string convert_rois_lib_path = "/usr/local/neuware/lib64/libmagicmind_plugin.so";
  auto convert_rois_kernel_lib = dlopen(convert_rois_lib_path.c_str(),RTLD_LAZY);
  NNADAPTER_CHECK(convert_rois_kernel_lib) << "Failed to dlopen " << convert_rois_lib_path;

  Clear();
  if (model && cache->dir && cache->token) {
    dump_graph_path_ = string_format("%s/%s.dat", cache->dir, cache->token);
  }
  dump_graph_buffer_ = &cache->buffer;
  if (cache->buffer.empty()) {
    NNADAPTER_CHECK_EQ(BuildFromModel(model), NNADAPTER_NO_ERROR);
    if (cache->dir) {
      cache->buffer = model_buffer_;
    }
  } else {
    NNADAPTER_CHECK_EQ(BuildFromCache(cache), NNADAPTER_NO_ERROR);
  }
  mm_engine_.reset(mm_model_->CreateIEngine());
  mm_context_.reset(mm_engine_->CreateIContext());
  if(GetBoolFromEnv("DUMP_MM_TENSOR")){
    NNADAPTER_LOG(WARNING) << "DUMP_MM_TENSOR is ON all tensors in magicmind will dump to mlu_mm_dump_tensors dir";
    magicmind::ContextDumpInfo dump_info;
    dump_info.SetDumpMode(magicmind::ContextDumpInfo::DumpMode::kAllTensors);
    dump_info.SetPath("mlu_mm_dump_tensors"); 
    dump_info.SetFileFormat(magicmind::ContextDumpInfo::FileFormat::kText);
    mm_context_->SetContextDumpInfo(dump_info);
  }
  enable_mm_profile_ = GetBoolFromEnv("ENABLE_MM_PROFILE");
  if (enable_mm_profile_){
    std::string profile_log_dir = "mm_profile_log";
    NNADAPTER_LOG(WARNING) << "ENABLE_MM_PROFILE is set true, magicmind profile log will write to mm_profile_log dir";
    magicmind::ProfilerOptions options;
    options.SetHostTracerLevel(magicmind::HostTracerLevel::kVerbose);
    options.SetDeviceTracerLevel(magicmind::DeviceTracerLevel::kOn);
    profiler_ = magicmind::CreateIProfiler(options,profile_log_dir.c_str());
    profiler_->Start();
    NNADAPTER_LOG(WARNING) << "magicmind profile start...";
  }else{
    NNADAPTER_LOG(WARNING) << "ENABLE_MM_PROFILE is false";
  }
  return NNADAPTER_NO_ERROR;
}

int Program::BuildFromCache(core::Cache* cache) {

  input_types_ = cache->input_types;
  output_types_ = cache->output_types;
  auto input_count = cache->input_types.size();
  size_t buffer_size = cache->buffer.size();
  size_t version_size = sizeof(float);
  size_t perm_size = input_count * sizeof(int);
  size_t model_size = buffer_size - version_size - perm_size;
  memcpy(&model_version_, &cache->buffer[0], version_size);
  mm_model_.reset(magicmind::CreateIModel());
  MLU_MM_CHECK(mm_model_->DeserializeFromMemory(
      cache->buffer.data() + version_size, model_size));
  inputs_perm_.resize(input_count);
  memcpy(inputs_perm_.data(),
         &cache->buffer[model_size + version_size],
         perm_size);
  return NNADAPTER_NO_ERROR;
}

int Program::BuildFromModel(core::Model* model) {
  Clear();
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  FuseMatMulAddIntoFullyConnected(model);
  FixQuantizedOps(model);
  FixTranspose(model);
  ConstantFoldOperations(model);
  if(context_->get_fusion_yolobox_multiclass_nms3_to_detection_output()){
    FixNonMaxSuppression(model);
  }
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  std::stringstream op_params;
  if (!context_->op_params_file_path().empty()) {
    std::ifstream op_params_file(context_->op_params_file_path().c_str());
    if (!op_params_file.is_open()) {
      NNADAPTER_LOG(WARNING) << " op params file open failed.";
    } else {
      op_params << op_params_file.rdbuf();
    }
    op_params_file.close();
  }
  Converter converter(&tensors_, mm_network_.get(), op_params.str());
  NNADAPTER_LOG(INFO)
      << CAMBRICON_MLU_FUSION_YOLOBOX_MULTICLASS_NMS3_TO_DETECTION_OUTPUT
      << " is set to "
      << context_->get_fusion_yolobox_multiclass_nms3_to_detection_output();
  converter.set_fusion_yolobox_multiclass_nms3_to_detection_output(
      context_->get_fusion_yolobox_multiclass_nms3_to_detection_output());
  NNADAPTER_CHECK_EQ(converter.Apply(model), NNADAPTER_NO_ERROR);

  // Indentify the inputs and outputs
  auto input_count = model->input_operands.size();
  NNADAPTER_VLOG(3) << "Model input count: " << input_count;
  std::vector<magicmind::ITensor*> input_tensors;
  if (input_count > 0) {
    input_tensors.resize(input_count);
    input_types_.resize(input_count);
    input_names_.resize(input_count);
    for (size_t i = 0; i < input_count; i++) {
      auto operand = model->input_operands[i];
      const auto& type = operand->type;
      NNADAPTER_CHECK(tensors_.find(operand) != tensors_.end());
      input_tensors[i] = tensors_[operand].front();
      NNADAPTER_CHECK(input_tensors[i]);
      input_types_[i] = type;
      input_names_[i] = input_tensors[i]->GetTensorName();
    }
  }
  auto output_count = model->output_operands.size();
  NNADAPTER_VLOG(3) << "Model output count: " << output_count;
  NNADAPTER_CHECK_GT(output_count, 0);
  std::vector<magicmind::ITensor*> output_tensors(output_count);
  output_types_.resize(output_count);
  for (size_t i = 0; i < output_count; i++) {
    auto operand = model->output_operands[i];
    const auto& type = operand->type;
    NNADAPTER_CHECK(tensors_.find(operand) != tensors_.end());
    output_tensors[i] = tensors_[operand].back();
    mm_network_->MarkOutput(tensors_[operand].back());
    NNADAPTER_CHECK(output_tensors[i]);
    output_types_[i] = type;
  }
  // Get the info of inputs and outputs, and check the count and buffer size of
  // inputs and outputs
  int num_inputs = mm_network_->GetInputCount();
  NNADAPTER_CHECK_EQ(input_count, num_inputs);
  int num_outputs = mm_network_->GetOutputCount();
  NNADAPTER_CHECK_EQ(output_count, num_outputs);

  magicmind::IBuilderConfig* config = mm_builder_config_.get();
  magicmind::INetwork* network = mm_network_.get();
  if (context_->build_config_file_path().empty()) {
    config->ParseFromString(R"({"graph_shape_mutable": true})");
    config->ParseFromString(
        R"({"precision_config": {"precision_mode": "force_float32"}})");
  } else {
    config->ParseFromFile(context_->build_config_file_path());
  }
  mm_model_.reset(mm_builder_->BuildModel("camb_model", network, config));
  NNADAPTER_VLOG(3) << "Build success.";
  inputs_perm_.resize(input_count);
  for (size_t i = 0; i < input_count; i++) {
    inputs_perm_[i] = mm_model_->GetInputIndexByName(input_names_[i].c_str());
  }
  size_t model_size = 0;
  MLU_MM_CHECK(mm_model_->GetSerializedModelSize(&model_size));
  size_t version_size = sizeof(float);
  size_t inputs_perm_size = input_count * sizeof(int);
  model_buffer_.resize(version_size + model_size + inputs_perm_size);
  memcpy(&model_buffer_[0], &model_version_, version_size);
  MLU_MM_CHECK(mm_model_->SerializeToMemory(model_buffer_.data() + version_size,
                                            model_size));
     
  if(GetBoolFromEnv("DUMP_MM_MODEL", false)){
    NNADAPTER_LOG(WARNING) << "DUMP_MM_MODEL is ON, magicmind model will save to mm_model";
    MLU_MM_CHECK(mm_model_->SerializeToFile("mm_model"));
  }
  memcpy(&model_buffer_[version_size + model_size],
         inputs_perm_.data(),
         inputs_perm_size);
  return NNADAPTER_NO_ERROR;
}

int Program::CheckInputsAndOutputs(uint32_t input_count,
                                   core::Argument* input_arguments,
                                   uint32_t output_count,
                                   core::Argument* output_arguments) {
  // Check inputs
  for (uint32_t i = 0; i < input_count; i++) {
    // Get actual type
    auto& arg = input_arguments[i];
    NNAdapterOperandType type;
    arg.access(arg.memory, &type, nullptr);
    // Check dimensions count
    uint32_t count = type.dimensions.count;
    int32_t* data = type.dimensions.data;
    auto& src_dimensions = input_types_[i].dimensions;
    int32_t* src_data = src_dimensions.data;
    if (count != src_dimensions.count) {
      return NNADAPTER_INVALID_DIMENSIONS;
    }
    // Check dimensions data
    for (uint32_t j = 0; j < count; j++) {
      if (data[j] != src_data[j]) {
        return NNADAPTER_INVALID_DIMENSIONS;
      }
    }
  }
  return NNADAPTER_NO_ERROR;
}

int Program::Execute(uint32_t input_count,
                     core::Argument* input_arguments,
                     uint32_t output_count,
                     core::Argument* output_arguments) {
  int ret = CheckInputsAndOutputs(
      input_count, input_arguments, output_count, output_arguments);
  if (ret != NNADAPTER_NO_ERROR) return ret;
  NNADAPTER_VLOG(3) << "Execute begining.";
  std::vector<magicmind::IRTTensor*> inputs = {};
  std::vector<magicmind::IRTTensor*> outputs = {};
  std::vector<bool> need_free(input_count);
  MLU_MM_CHECK(magicmind::CreateInputTensors(mm_context_.get(), &inputs));
  for (uint32_t i = 0; i < input_count; i++) {
    void* ptr = nullptr;
    auto& arg = input_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, input_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = input_types_[arg.index];
    auto buffer = arg.access(arg.memory, &type, nullptr);
    NNADAPTER_CHECK(buffer);
    auto input_tensor = inputs.at(inputs_perm_[i]);
    if (IsDeviceMemory(input_tensor) && !IsDevicePtr(buffer)) {
      auto length = GetOperandTypeBufferLength(type);
      MLU_CNRT_CHECK(cnrtMalloc(&ptr, length));
      MLU_CNRT_CHECK(
          cnrtMemcpy(ptr, buffer, length, CNRT_MEM_TRANS_DIR_HOST2DEV));
      need_free.at(i) = true;
    } else if ((IsDeviceMemory(input_tensor) && IsDevicePtr(buffer)) ||
               (!IsDeviceMemory(input_tensor) && !IsDevicePtr(buffer))) {
      ptr = buffer;
      need_free.at(i) = false;
    } else {
      NNADAPTER_VLOG(3) << " Unsupport position : input_tensor is in device ? " << IsDeviceMemory(input_tensor) << " and tensor buffer is in device ? " << IsDevicePtr(buffer);
      ptr = buffer;
      need_free.at(i) = false;
    }
    input_tensor->SetData(ptr);
    input_tensor->SetDimensions(
        ConvertToMagicMindDims(type.dimensions.data, type.dimensions.count));
  }

  if(enable_mm_profile_){
    // FIXME(zhaoying): batch size is set to 1 just for now
    profiler_->StepBegin(1);
    NNADAPTER_LOG(WARNING) << "magicmind profile step begin...";
  }
  MLU_MM_CHECK(mm_context_->Enqueue(inputs, &outputs, queue_));
  MLU_CNRT_CHECK(cnrtQueueSync(queue_));
  if(enable_mm_profile_){
    profiler_->StepEnd(); 
    NNADAPTER_LOG(WARNING) << "magicmind profile step end...";
  }
  NNADAPTER_VLOG(3) << "Execute ending.";
  for (uint32_t i = 0; i < output_count; i++) {
    auto& arg = output_arguments[i];
    NNADAPTER_CHECK_GE(arg.index, 0);
    NNADAPTER_CHECK_LT(arg.index, output_count);
    NNADAPTER_CHECK(arg.memory);
    NNADAPTER_CHECK(arg.access);
    auto type = &output_types_[arg.index];
    auto out_dims = outputs[i]->GetDimensions();
    type->dimensions.data[0] = IsScalar(out_dims) ? 1 : out_dims[0];
    auto buffer = arg.access(arg.memory, type, nullptr);
    NNADAPTER_CHECK(buffer);
    void* output_mlu_ptr = outputs[i]->GetMutableData();
    if (IsDeviceMemory(outputs[i])) {
      MLU_CNRT_CHECK(cnrtMemcpy(buffer,
                                output_mlu_ptr,
                                outputs[i]->GetSize(),
                                CNRT_MEM_TRANS_DIR_DEV2HOST));
    } else {
      memcpy(buffer, output_mlu_ptr, outputs[i]->GetSize());
    }
  }

  for (uint32_t i = 0; i < input_count; i++) {
    auto input = inputs.at(i);
    if (need_free[i]) {
      MLU_CNRT_CHECK(cnrtFree(input->GetMutableData()));
    }
    input->Destroy();
  }
  for (auto output : outputs) {
    output->Destroy();
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
