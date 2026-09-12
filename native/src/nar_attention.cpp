#include "lyra/nar.hpp"
#include "lyra/nar_attention.hpp"
#include <mlx/backend/metal/metal.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <regex>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

// Cooperative-tensor lane layout adapted from MLX's steel/attn/nax.h.
// Copyright © 2023-2025 Apple Inc. (MIT License)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
namespace lyra {
bool mpp_attention_available() {
  static const bool available = [] {
#ifdef __APPLE__
    if (!mx::metal::is_available()) return false;
    char version[64]{};
    size_t size = sizeof(version);
    if (sysctlbyname("kern.osproductversion", version, &size, nullptr, 0) != 0) return false;
    int major = 0, minor = 0;
    if (std::sscanf(version, "%d.%d", &major, &minor) < 1 || major < 26 || (major == 26 && minor < 2)) return false;
    const auto& info = mx::metal::device_info();
    auto it = info.find("architecture");
    if (it == info.end() || !std::holds_alternative<std::string>(it->second)) return false;
    std::smatch match;
    const auto& architecture = std::get<std::string>(it->second);
    if (!std::regex_match(architecture, match, std::regex("applegpu_g([0-9]+)([a-z])"))) return false;
    return std::stoi(match[1].str()) >= (match[2].str() == "p" ? 18 : 17);
#else
    return false;
#endif
  }();
  return available;
}
namespace {
mx::array mpp_attention(const mx::array& query, const mx::array& key, const mx::array& value) {
  static const auto kernel = [] {
    mx::CompileOptions options;
    options.math_mode = mx::MathMode::Safe;
    return mx::fast::metal_kernel("lyra_nar_precise_attention", {"query", "key", "value"}, {"output"}, R"METAL(
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
)METAL", "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n", true, false, options);
  }();
  const int length = query.shape(2), heads = query.shape(1);
  return kernel({query,key,value}, {query.shape()}, {mx::bfloat16},
    {((length+63)/64)*128,heads,query.shape(0)}, {128,1,1},
    {{"T",mx::bfloat16},{"HQ",heads},{"HK",key.shape(1)},{"QL",length},{"KL",key.shape(2)}},
    std::nullopt, false, {})[0];
}
mx::array tokens(const mx::array& x, int begin, int end) {
  return mx::slice(x,{0,0,begin,0},{x.shape(0),x.shape(1),end,x.shape(3)});
}
}
mx::array nar_attention(const mx::array& query, const mx::array& key, const mx::array& value,
                        bool causal, int query_chunk_size) {
  if (query.ndim()!=4 || key.ndim()!=4 || value.shape()!=key.shape())
    throw Error("ValueError","Expected attention Q/K/V shaped [batch,heads,tokens,dim]");
  if (query.shape(0)!=key.shape(0) || query.shape(3)!=key.shape(3))
    throw Error("ValueError","Attention batch and head dimensions do not match");
  if (key.shape(1)<1 || query.shape(1)<1 || query.shape(1)%key.shape(1))
    throw Error("ValueError","Invalid grouped-query head count");
  if (query.shape(2)<1 || key.shape(2)<1) throw Error("ValueError","Attention sequences must be nonempty");
  if (causal && query.shape(2)!=key.shape(2)) throw Error("ValueError","Causal prefill requires equal query and key lengths");
  if (query_chunk_size<1) throw Error("ValueError","query_chunk_size must be a positive integer");
  if (!causal && query.dtype()==mx::bfloat16 && key.dtype()==mx::bfloat16 && value.dtype()==mx::bfloat16 &&
      query.shape(3)==128 && query.shape(2)>8 && query_chunk_size>=64 && mpp_attention_available())
    return mpp_attention(query,key,value);
  const bool precise = causal || std::min(query_chunk_size,query.shape(2))>8;
  auto precise_key = precise ? mx::astype(key,mx::float32) : key;
  auto precise_value = precise ? mx::astype(value,mx::float32) : value;
  std::vector<mx::array> outputs;
  outputs.reserve(1+(query.shape(2)-1)/query_chunk_size);
  const float scale = static_cast<float>(std::pow(query.shape(3),-0.5));
  for (int start=0;start<query.shape(2);) {
    int end=start+std::min(query_chunk_size,query.shape(2)-start);
    auto used_key = causal ? tokens(precise_key,0,end) : end-start<=8 ? key : precise_key;
    auto used_value = causal ? tokens(precise_value,0,end) : end-start<=8 ? value : precise_value;
    std::optional<mx::array> mask;
    if (causal) mask=mx::less_equal(mx::reshape(mx::arange(end),{1,end}),mx::reshape(mx::arange(start,end),{end-start,1}));
    outputs.push_back(model_ops::sdpa(tokens(query,start,end),used_key,used_value,scale,mask));
    start=end;
  }
  return outputs.size()==1 ? outputs.front() : mx::concatenate(outputs,2);
}
}
