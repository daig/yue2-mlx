"""Full-key BF16 acoustic attention with FP32 probabilities and accumulation."""

# Cooperative-tensor lane layout adapted from MLX's steel/attn/nax.h.
# Copyright © 2023-2025 Apple Inc. (MIT License)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from functools import cache
import platform
import re

import mlx.core as mx


@cache
def is_available() -> bool:
    """Match MLX's NAX hardware/OS gate before compiling MPP instructions."""
    if (
        platform.system() != "Darwin"
        or tuple(map(int, platform.mac_ver()[0].split(".")[:2])) < (26, 2)
        or not mx.metal.is_available()
    ):
        return False
    architecture = re.fullmatch(r"applegpu_g(\d+)([a-z])", mx.device_info()["architecture"])
    return architecture is not None and int(architecture[1]) >= (
        18 if architecture[2] == "p" else 17
    )


# Fixed 64-query/16-key tiles bound register storage independently of song length.
# Safe math and relaxed_precision=false are required for the FP32-left P@V.
# Explicit contiguous load groups allow Metal to vectorize BF16 operand reads.
_SOURCE = r"""
constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
    16, 16, 32, false, true, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
    16, 32, 16, false, false, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
mpp::tensor_ops::matmul2d<pv_desc, metal::execution_simdgroup> pv_op;
auto qa = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
auto kb = qk_op.template get_right_input_cooperative_tensor<T, T, float>();
auto scores = qk_op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(qa)>, metal::remove_addrspace_t<decltype(kb)>, float>();
auto pa = pv_op.template get_left_input_cooperative_tensor<float, T, float>();
auto vb = pv_op.template get_right_input_cooperative_tensor<float, T, float>();
auto accum = pv_op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(vb)>, float>();
const uint q_start = threadgroup_position_in_grid.x * 64 + simdgroup_index_in_threadgroup * 16;
const uint h = threadgroup_position_in_grid.y;
const uint b = threadgroup_position_in_grid.z;
const uint q_base = (b * HQ + h) * QL * 128;
const uint kv_base = (b * HK + h / (HQ / HK)) * KL * 128;
const uint lane_id = thread_index_in_simdgroup;
const uint sr = ((lane_id >> 1) & 3) | ((lane_id >> 2) & 4);
const uint sc = ((lane_id & 1) << 2) | (lane_id & 8);
const uint vc = ((lane_id & 1) << 1) | ((lane_id & 8) >> 1);
float output_accum[4][16] = {};
float row_max[2] = {-INFINITY, -INFINITY};
float row_sum[2] = {0.0f, 0.0f};
constexpr float scale2 = 0.08838834764831845f * 1.4426950408889634f;
for (uint key_start = 0; key_start < KL; key_start += 16) {
    #pragma unroll
    for (uint i = 0; i < scores.get_capacity(); ++i) scores[i] = 0.0f;
    #pragma clang loop unroll_count(4)
    for (uint depth = 0; depth < 128; depth += 32) {
        #pragma unroll
        for (uint r = 0; r < 2; ++r) {
            const uint row = q_start + sr + r * 8;
            const device T* ptr = query + q_base + row * 128 + depth + sc;
            #pragma unroll
            for (uint block = 0; block < 2; ++block) {
                #pragma unroll
                for (uint c = 0; c < 4; ++c)
                    qa[block * 8 + r * 4 + c] = row < QL ? ptr[block * 16 + c] : T(0);
            }
        }
        #pragma unroll
        for (uint r = 0; r < 2; ++r) {
            const uint row = key_start + sr + r * 8;
            const device T* ptr = key + kv_base + row * 128 + depth + sc;
            #pragma unroll
            for (uint block = 0; block < 2; ++block) {
                #pragma unroll
                for (uint c = 0; c < 4; ++c)
                    kb[block * 8 + r * 4 + c] = row < KL ? ptr[block * 16 + c] : T(0);
            }
        }
        qk_op.run(qa, kb, scores);
    }
    float new_max[2] = {row_max[0], row_max[1]};
    #pragma unroll
    for (uint i = 0; i < scores.get_capacity(); ++i) {
        const uint col = sc + (i % 4);
        scores[i] = key_start + col < KL ? scores[i] * scale2 : -INFINITY;
        uint row = i / 4;
        new_max[row] = metal::max(new_max[row], scores[i]);
    }
    #pragma unroll
    for (uint row = 0; row < 2; ++row) {
        new_max[row] = metal::max(new_max[row], simd_shuffle_xor(new_max[row], ushort(1)));
        new_max[row] = metal::max(new_max[row], simd_shuffle_xor(new_max[row], ushort(8)));
    }
    #pragma unroll
    for (uint i = 0; i < scores.get_capacity(); ++i) {
        uint row = i / 4;
        scores[i] = metal::fast::exp2(scores[i] - new_max[row]);
    }
    float factor[2];
    #pragma unroll
    for (uint row = 0; row < 2; ++row) {
        // Match the native 8-key fragment's pairwise reduction order.
        float block_sum = (scores[row * 4] + scores[row * 4 + 1]) +
            (scores[row * 4 + 2] + scores[row * 4 + 3]);
        block_sum += simd_shuffle_xor(block_sum, ushort(1));
        block_sum += simd_shuffle_xor(block_sum, ushort(8));
        factor[row] = metal::fast::exp2(row_max[row] - new_max[row]);
        row_sum[row] = row_sum[row] * factor[row] + block_sum;
        row_max[row] = new_max[row];
    }
    #pragma unroll
    for (uint dim_block = 0; dim_block < 4; ++dim_block) {
        if (dim_block == 2) threadgroup_barrier(mem_flags::mem_none);
        #pragma unroll
        for (uint i = 0; i < accum.get_capacity(); ++i)
            accum[i] = output_accum[dim_block][i] * factor[i / 8];
        #pragma unroll
        for (uint i = 0; i < pa.get_capacity(); ++i) {
            // Precise FP32-left operands pack two columns, not four.
            const uint col = vc + ((i % 4) / 2) * 8 + (i % 2);
            const uint lane = (thread_index_in_simdgroup & ~9u) |
                ((col & 4u) >> 2) | (col & 8u);
            const uint index = (i / 4) * 4 + (i % 2);
            const float low = simd_shuffle(scores[index], lane);
            const float high = simd_shuffle(scores[index + 2], lane);
            pa[i] = (col & 2u) ? high : low;
        }
        #pragma unroll
        for (uint r = 0; r < 2; ++r) {
            const uint row = key_start + sr + r * 8;
            const device T* ptr = value + kv_base + row * 128 + dim_block * 32 + vc;
            #pragma unroll
            for (uint group = 0; group < 4; ++group) {
                #pragma unroll
                for (uint c = 0; c < 2; ++c)
                    vb[r * 8 + group * 2 + c] = row < KL ? ptr[group * 8 + c] : T(0);
            }
        }
        pv_op.run(pa, vb, accum);
        #pragma unroll
        for (uint i = 0; i < accum.get_capacity(); ++i)
            output_accum[dim_block][i] = accum[i];
    }
}
#pragma unroll
for (uint dim_block = 0; dim_block < 4; ++dim_block) {
    #pragma unroll
    for (uint i = 0; i < accum.get_capacity(); ++i) {
        const uint col = vc + ((i % 8) / 2) * 8 + (i % 2);
        const uint row = q_start + sr + (i / 8) * 8;
        if (row < QL)
            output[q_base + row * 128 + dim_block * 32 + col] =
                T(output_accum[dim_block][i] / row_sum[i / 8]);
    }
}
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="lyra_nar_precise_attention",
        input_names=["query", "key", "value"],
        output_names=["output"],
        source=_SOURCE,
        header="#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n",
        compile_options={"math_mode": "safe"},
    )


def attention(query: mx.array, key: mx.array, value: mx.array) -> mx.array:
    """Validated noncausal BF16 GQA, D=128; all queries see all keys."""
    batch, heads, length, _ = query.shape
    (result,) = _kernel()(
        inputs=[query, key, value],
        template=[
            ("T", mx.bfloat16),
            ("HQ", heads),
            ("HK", key.shape[1]),
            ("QL", length),
            ("KL", key.shape[2]),
        ],
        grid=(((length + 63) // 64) * 128, heads, batch),
        threadgroup=(128, 1, 1),
        output_shapes=[query.shape],
        output_dtypes=[mx.bfloat16],
    )
    return result
