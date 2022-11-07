// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "driver/cambricon_mlu/optimizer/fix_transpose.h"
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace cambricon_mlu {

static bool NeedDeleteTranspose(core::Model* model, core::Operand* perm_operand) {
  auto perm_count = perm_operand->length / sizeof(int32_t);
  auto perm_data = reinterpret_cast<int32_t*>(perm_operand->buffer);
  bool need_delete = true;
  for (uint32_t i = 0; i < perm_count; i++) {
    if (perm_data[i] != i) {
      need_delete = false;
    }
  }
  return need_delete;
}

// Del a dequant operation after output_operand
static void DelTransposeOperation(core::Model* model,
                                  core::Operation* transpose_operation) {
  auto input_operands = transpose_operation->input_operands;
  auto input_operand = input_operands[0];
  auto perm_operand = input_operands[1];
  auto output_operand = transpose_operation->output_operands[0];
  auto next_operations = GetOperandConsumers(model, output_operand);
  for (auto operation : next_operations) {
    for (int i = 0; i < operation->input_operands.size(); i++) {
      if (output_operand == operation->input_operands[i]) {
        operation->input_operands[i] = input_operand;
      }
    }
  }
  RemoveOperand(model, perm_operand);
  RemoveOperand(model, output_operand);
  RemoveOperation(model, transpose_operation);
}

static void DeleteTranspose(core::Model* model) {
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    auto input_operands = operation->input_operands;
    if (operation->type != NNADAPTER_TRANSPOSE) {
      continue;
    }
    auto output_operand = operation->output_operands[0];
    if (NeedDeleteTranspose(model, input_operands[1])) {
      DelTransposeOperation(model, operation);
    }
  }
}

void FixTranspose(core::Model* model) {
  DeleteTranspose(model);
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
