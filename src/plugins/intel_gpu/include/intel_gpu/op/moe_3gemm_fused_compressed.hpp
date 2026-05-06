// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ov_ops/moe_compressed.hpp"

namespace ov::intel_gpu::op {

/// \brief MOE3GemmFusedCompressed that support compressed and fused MOE for GEMM3_SWIGLU.
class MOE3GemmFusedCompressed : public ov::op::internal::MOECompressed {
public:
    OPENVINO_OP("MOE3GemmFusedCompressed", "gpu_opset", ov::op::internal::MOECompressed);

    MOE3GemmFusedCompressed() = default;

    /// \brief Constructs a MOE3GemmFusedCompressed operation with config only
    /// \param args The input tensors, in the following order:
    ///   0: hidden_states - input tensor with hidden representations
    ///   1: topk_weights - [num_tokens, top_k] pre-computed routing weights from MoERouterFused
    ///   2: w0_weight - expert weights for first projection
    ///   3: w0_scale
    ///   4: w0_zp
    ///   5: w1_weight - expert weights for second projection
    ///   6: w1_scale
    ///   7: w1_zp
    ///   8: w2_weight - expert weights for final projection
    ///   9: w2_scale
    ///   10: w2_zp
    ///   11: topk_indices - [num_tokens, top_k] pre-computed expert indices from MoERouterFused
    /// \param config Configuration for the MOE 3GEMM SWIGLU fused operation
    MOE3GemmFusedCompressed(const OutputVector& args, const MOECompressed::Config config);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

}  // namespace ov::intel_gpu::op
