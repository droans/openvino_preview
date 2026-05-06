# Task: Split MoERouterFused out of MOE3GemmFusedCompressed

## Background

The Intel GPU plugin fuses a 3-GEMM MoE block into a single
`MOE3GemmFusedCompressed` op through the following transformation chain:

```
ConvertTiledMoeBlockToGatherMatmuls
  → ConvertGatherMatmulToGatherMatmulCompressed
    → MoeOpFusion
      → FuseMOE3GemmCompressed    ← this is the final fusion step
```

`FuseMOE3GemmCompressed` pattern-matches **both** the routing subgraph (MatMul +
Softmax/Sigmoid+TopK+Normalize, two variants) **and** the `MOECompressed` op
(gate/up/down compressed-weight GEMMs), then replaces both with a single
`MOE3GemmFusedCompressed` node.  The current fused op therefore bundles two
logically distinct computations:

1. **Router**: logit projection (MatMul), expert selection (TopK), weight
   normalization (ReduceSum+Divide or GatherElements+Divide) — two routing
   variants: `SOFTMAX` and `SIGMOID_BIAS`.
2. **MoE FFN**: three compressed-weight GEMMs (gate, up, down) with SwiGLU,
   optional shared experts.

Because the routing logic is baked in, any change to the routing pattern
(e.g. a new routing variant) forces a full rewrite of the fused op.

**Goal**: Extract the routing computation into a new GPU-internal
`MoERouterFused` op/primitive/kernel, so that:
- `MoERouterFused` handles expert selection and weight normalization.
- `MOE3GemmFusedCompressed` handles only the FFN GEMMs (it now receives the
  router's outputs instead of raw logits).
- Routing variants can be added/changed without touching the GEMM kernel.

---

## Key files to read first (in order)

Before writing any code, read these files to understand the existing structure:

| File | Purpose |
|------|---------|
| `src/plugins/intel_gpu/src/plugin/transformations/fuse_moe_3gemm_compressed.cpp` | Current fusion pass — pattern matcher + callback |
| `src/plugins/intel_gpu/src/plugin/transformations/fuse_moe_3gemm_compressed.hpp` | Pass declaration |
| `src/plugins/intel_gpu/include/intel_gpu/op/moe_3gemm_fused_compressed.hpp` | GPU-internal op header |
| `src/plugins/intel_gpu/src/plugin/transformations/op/moe_3gemm_fused_compressed.cpp` | Op implementation |
| `src/plugins/intel_gpu/include/intel_gpu/primitives/moe_3gemm_fused_compressed.hpp` | cldnn primitive |
| `src/plugins/intel_gpu/src/graph/include/moe_3gemm_fused_inst.h` | Program node + inst template |
| `src/plugins/intel_gpu/src/graph/moe_3gemm_fused.cpp` | Inst calc_output_layouts |
| `src/plugins/intel_gpu/src/graph/registry/moe_3gemm_swiglu_impls.cpp` | Implementation registry entry |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.hpp` | OCL implementation class |
| `src/plugins/intel_gpu/src/plugin/ops/moe.cpp` | `CreateMOE3GemmFusedCompressedOp` builder |
| `src/common/transformations/include/ov_ops/moe_compressed.hpp` | `MOECompressed` base op + Config |
| `src/plugins/intel_gpu/tests/unit/transformations/fuse_moe_3gemm_compressed_test.cpp` | Unit tests for the fusion pass |
| `src/plugins/intel_gpu/tests/functional/subgraph_tests/dynamic/moe.cpp` | Functional tests |

---

## Step 0 — Investigate routing weight usage

**Before designing anything**, answer this question by reading the relevant
OCL kernel source files under
`src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/`:

> Does the `moe_3gemm_fused_compressed` kernel consume **normalized routing
> weights** (i.e. the post-divide floating-point scores per expert), or does
> it only consume the **top-k expert indices** (integers)?

- If the kernel **only** uses indices for token dispatch and does not need the
  normalized weights, then `MoERouterFused` should produce **one output**:
  expert indices `[num_tokens, topk]`.
- If the kernel **also** multiplies by routing weights at the scatter/reduce
  stage, then `MoERouterFused` should produce **two outputs**:
  - output 0: normalized routing weights `[num_tokens, topk]`
  - output 1: expert indices `[num_tokens, topk]`

Document your finding as a short comment at the top of the new op header.

---

## Step 1 — New op: `MoERouterFused`

Create the following files (modelled exactly on the `MOE3GemmFusedCompressed`
pattern):

### 1a. Op header
`src/plugins/intel_gpu/include/intel_gpu/op/moe_router_fused.hpp`

```cpp
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

/// \brief MoERouterFused — GPU-internal op that computes expert selection and
/// routing-weight normalization for Mixture-of-Experts blocks.
///
/// Inputs:
///   0: router_logits  [num_tokens, num_experts]  — output of the router MatMul
///   1: routing_bias   (optional, SIGMOID_BIAS only) [1, num_experts]
///   2: routing_eps    (optional, SIGMOID_BIAS only) scalar constant
///
/// Outputs (fill in the exact count based on Step 0 investigation):
///   0: expert_indices [num_tokens, topk]          — always present
///  [1: norm_weights   [num_tokens, topk]]          — only if weights are consumed
///
class MoERouterFused : public ov::op::Op {
public:
    OPENVINO_OP("MoERouterFused", "gpu_opset");

    struct Config {
        size_t num_expert = 0;
        size_t top_k      = 0;
        // Extend with any other fields the routing kernel needs.
        // Use ov::op::internal::MOECompressed::RoutingType for the enum.
    };

    MoERouterFused() = default;
    MoERouterFused(const ov::OutputVector& args, const Config& config);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    const Config& get_config() const { return m_config; }

private:
    Config m_config;
};

}  // namespace ov::intel_gpu::op
```

Implement the class in:
`src/plugins/intel_gpu/src/plugin/transformations/op/moe_router_fused.cpp`

Follow `moe_3gemm_fused_compressed.cpp` for the implementation pattern
(`validate_and_infer_types`, `clone_with_new_inputs`, `visit_attributes`).
Set the number of outputs and their shapes based on the Step 0 finding.

### 1b. Primitive header
`src/plugins/intel_gpu/include/intel_gpu/primitives/moe_router_fused.hpp`

Model it on `moe_3gemm_fused_compressed.hpp`:
- Carry the `MoERouterFused::Config` (not the full `MOECompressed::Config`).
- Input list in the doc comment must match the op inputs above.
- Use `CLDNN_DECLARE_PRIMITIVE(moe_router_fused)`.

---

## Step 2 — Program node + instance

### 2a. Instance header
`src/plugins/intel_gpu/src/graph/include/moe_router_fused_inst.h`

Model on `moe_3gemm_fused_inst.h`. Provide:
- `typed_program_node<moe_router_fused>` with `get_shape_infer_dependencies`
  returning `{}`.
- `typed_primitive_inst<moe_router_fused>` with `calc_output_layouts`,
  `calc_output_layout`, `to_string`.

### 2b. Instance implementation
`src/plugins/intel_gpu/src/graph/moe_router_fused.cpp`

Model on `moe_3gemm_fused.cpp`. Implement `calc_output_layouts` to produce the
correct output shapes derived from the input shapes and the config
(`num_tokens` × `topk` for indices; same for weights if applicable).

---

## Step 3 — OCL kernel stub

### 3a. Registry entry
`src/plugins/intel_gpu/src/graph/registry/moe_router_fused_impls.cpp`

Model on `moe_3gemm_swiglu_impls.cpp`. Register a single OCL
`ImplementationManager` for `moe_router_fused`.

### 3b. OCL implementation stub
`src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_router_fused_opt.hpp`
`src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_router_fused_opt.cpp`

Create a minimal stub that compiles (inherits from the appropriate OCL base,
registers with `REGISTER_DEFAULT_IMPL`).  You do **not** need a working kernel
body — a TODO comment is acceptable.  The purpose is to ensure the full
build path (registry → impl → OCL) compiles without errors.

---

## Step 4 — Op registration in `program_builder`

In `src/plugins/intel_gpu/src/plugin/ops/moe.cpp`:

1. Add `#include "intel_gpu/op/moe_router_fused.hpp"` and
   `#include "intel_gpu/primitives/moe_router_fused.hpp"`.

2. Implement `CreateMoERouterFusedOp`:
   ```cpp
   static void CreateMoERouterFusedOp(
       ProgramBuilder& p,
       const std::shared_ptr<ov::intel_gpu::op::MoERouterFused>& op)
   {
       auto inputs = p.GetInputInfo(op);
       validate_inputs_count(op, {/* 1 for SOFTMAX, 3 for SIGMOID_BIAS */});
       const std::string layerName = layer_type_name_ID(op);
       p.add_primitive(*op, cldnn::moe_router_fused(layerName, inputs, op->get_config()));
   }
   ```
   Set the expected input counts based on the routing variant (check
   `op->get_config().routing_type`).

3. Add `REGISTER_FACTORY_IMPL(internal, MoERouterFused);` alongside the
   existing `REGISTER_FACTORY_IMPL` lines.

   > **Note**: `REGISTER_FACTORY_IMPL(internal, X)` maps to
   > `ov::op::internal::X`.  Because `MoERouterFused` is in
   > `ov::intel_gpu::op`, you may need to add a type alias in the `ov::op::internal`
   > namespace (see how `MOE3GemmFusedCompressed` does it at the top of `moe.cpp`).

---

## Step 5 — Refactor `FuseMOE3GemmCompressed`

This is the core transformation change.

### 5a. What the callback currently does

The callback in `fuse_moe_3gemm_compressed.cpp`:
1. Collects pattern-matched nodes (routing: matmul, softmax/sigmoid, topk,
   divide; FFN: moe_compressed + compressed weight constants).
2. Builds one `MOE3GemmFusedCompressed` node with all inputs: hidden_state,
   router_matmul output, weight tensors, optionally routing_bias + routing_eps.
3. Replaces `MOECompressed` with the fused node.

### 5b. New callback logic

Split the single node creation into two:

**Create `MoERouterFused`** from:
- Input 0: the router MatMul output (`pattern_map.at(matmul)`)
- Input 1 (SIGMOID_BIAS only): routing_bias constant
- Input 2 (SIGMOID_BIAS only): routing_eps constant

Populate `MoERouterFused::Config` from the matched config fields:
`num_expert`, `top_k`, `routing_type`.

**Create `MOE3GemmFusedCompressed`** with updated inputs:
- Input 0: `hs_reshaped` (hidden states — unchanged)
- Input 1: output of `MoERouterFused` that carries normalized weights (if
  applicable per Step 0); otherwise the expert-indices output
- Input 2: expert indices output of `MoERouterFused` (if weights are a
  separate output; drop this slot if the router only produces indices)
- Inputs 3–N: compressed weight tensors (unchanged)

> The exact rewiring depends on whether `MOE3GemmFusedCompressed` needs routing
> weights, routing indices, or both.  Use the Step 0 finding to decide.

### 5c. `ov::copy_runtime_info` and naming

Call `ov::copy_runtime_info` on both new nodes.  Set the friendly name of
`MOE3GemmFusedCompressed` to the original `moe_compressed` friendly name.

### 5d. Validate-and-infer-types for `MOE3GemmFusedCompressed`

Update `MOE3GemmFusedCompressed::validate_and_infer_types` and `moe.cpp`'s
`CreateMOE3GemmFusedCompressedOp` to reflect the new input layout (the routing
bias/eps inputs are gone; the first routing input now comes from
`MoERouterFused`).  Keep the shared-expert slots unchanged.

---

## Step 6 — CMakeLists updates

Add the new source files to the relevant `CMakeLists.txt` targets:
- `src/plugins/intel_gpu/src/plugin/transformations/CMakeLists.txt` — for the
  new op implementation.
- `src/plugins/intel_gpu/src/graph/CMakeLists.txt` — for `moe_router_fused.cpp`.
- `src/plugins/intel_gpu/src/graph/registry/CMakeLists.txt` or the impls
  target — for `moe_router_fused_impls.cpp`.
- The OCL v2 impls target — for `moe_router_fused_opt.cpp`.

---

## Step 7 — Update `keep_moe_3gemm_const_precision`

Read `src/plugins/intel_gpu/src/plugin/transformations/keep_moe_3gemm_const_precision.cpp`.
This pass walks the inputs of `MOE3GemmFusedCompressed` to tag constant
precision.  After the split, routing constants (bias, eps) are inputs to
`MoERouterFused`, not `MOE3GemmFusedCompressed`.  Update this pass accordingly.

---

## Step 8 — Update tests

### 8a. Unit test
`src/plugins/intel_gpu/tests/unit/transformations/fuse_moe_3gemm_compressed_test.cpp`

After the transformation the graph should contain:
- 1 × `MoERouterFused` node
- 1 × `MOE3GemmFusedCompressed` node (with `MoERouterFused` output as a routing input)

Update the assertion that previously checked for a single
`MOE3GemmFusedCompressed` to also require a `MoERouterFused`.

### 8b. Functional test
`src/plugins/intel_gpu/tests/functional/subgraph_tests/dynamic/moe.cpp`

In `MoECompressedFusionTest::validate()`, update the GEMM3 branch:

```cpp
// Before:
ov::test::CheckNumberOfNodesWithType(compiledModel, "moe_3gemm_fused_compressed", 1);

// After:
ov::test::CheckNumberOfNodesWithType(compiledModel, "moe_3gemm_fused_compressed", 1);
ov::test::CheckNumberOfNodesWithType(compiledModel, "moe_router_fused", 1);
```

---

## Constraints and conventions

- **No functional regression**: the split must not change inference results.
  The two new ops together must compute exactly what the single old op did.
- **Copyright header**: `// Copyright (C) 2018-2026 Intel Corporation`
- **Namespace**: new op lives in `ov::intel_gpu::op`; primitive in `cldnn`.
- **No `using namespace` in headers**.
- Do not modify the GEMM2 path (`CreateMOECompressedOp` / `GEMM2_BIAS_SWIGLU_CLAMP`).
- Do not modify `ConvertTiledMoeBlockToGatherMatmuls` or earlier passes.
- Keep `MOECompressed::Config` unchanged (do not add `RouterConfig` fields to
  it); `MoERouterFused::Config` is a separate, minimal struct.
- Prefer `ov::op::v*::OpName` namespace style in new code.

---

## Deliverable checklist

When done, verify that all of the following hold:

- [ ] The project compiles without errors (`cmake --build build --target ov_gpu_func_tests -j$(nproc)`).
- [ ] `./bin/intel64/Release/ov_gpu_func_tests --gtest_filter=*smoke_MoE3Gemm*` passes.
- [ ] The validate() in `MoECompressedFusionTest` confirms both
      `moe_3gemm_fused_compressed` (×1) and `moe_router_fused` (×1) are
      present in the compiled model.
- [ ] The GEMM2 fusion test (`*smoke_MoE2Gemm*`) still passes unchanged.
- [ ] `./bin/intel64/Release/ov_gpu_unit_tests --gtest_filter=*moe_3gemm*` passes.
