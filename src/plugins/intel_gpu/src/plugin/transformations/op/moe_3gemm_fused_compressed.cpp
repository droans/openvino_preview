// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/moe_compressed.hpp"
#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "ov_ops/moe_compressed.hpp"

namespace ov::intel_gpu::op {

MOE3GemmFusedCompressed::MOE3GemmFusedCompressed(const OutputVector& args, const ov::op::internal::MOECompressed::Config config) : ov::op::internal::MOECompressed() {
    m_config = config;
    set_arguments(args);
    constructor_validate_and_infer_types();
}

void MOE3GemmFusedCompressed::validate_and_infer_types() {
    // Input layout: [hs, topk_weights, w0..zp2, topk_indices, (shared_*)?]
    const size_t expected_inputs = m_config.num_shared_expert > 0 ? 22 : 12;
    OPENVINO_ASSERT(get_input_size() == expected_inputs,
                    "MOE3GemmFusedCompressed: expected ",
                    expected_inputs,
                    " inputs, got ",
                    get_input_size());

    // Output shape = hidden_states shape
    auto output_type = m_config.out_type == ov::element::dynamic ? get_input_element_type(0) : m_config.out_type;
    set_output_type(0, output_type, get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> MOE3GemmFusedCompressed::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<MOE3GemmFusedCompressed>(new_args, get_config());
}

}  // namespace ov::intel_gpu::op
