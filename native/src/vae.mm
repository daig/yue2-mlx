/*
 * Decoder topology and SnakeBeta derived from stable-audio-tools
 * a6ae0cdf8b2eb1567a4b42ceadddec3712d99d45 via vendor/yue/modeling_vae.py.
 * Copyright (c) 2023 Stability AI
 * Copyright (c) 2022 NVIDIA CORPORATION
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Weight normalization follows PyTorch 2.10's MPS WeightNorm.mm operator order.
 * PyTorch's BSD copyright/license notice is reproduced in noise.cpp.
 */
#include "lyra/vae.hpp"
#include "lyra/storage.hpp"
#include "lyra/runtime.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <set>

namespace lyra {
namespace {
void check_cancel(const Cancelled& cancelled) {
  if (cancelled && cancelled()) throw Error("InterruptedError", "Cancelled during VAE decoding");
}
MPSShape* shape(const std::vector<int64_t>& dims) {
  NSMutableArray<NSNumber*>* result = [NSMutableArray arrayWithCapacity:dims.size()];
  for (auto d : dims) [result addObject:@(d)];
  return result;
}
struct Tensor {
  std::vector<int64_t> dims;
  MPSGraphTensorData* data = nil;
};
// Decoder-only, seek-based safetensors reader: encoder payloads are never read.
struct TensorSource {
  fs::path file;
  uint64_t offset, bytes;
  std::vector<int64_t> dims;
};
std::map<std::string, TensorSource> tensor_sources(const fs::path& directory) {
  std::set<std::string> files;
  if (fs::exists(directory / "model.safetensors.index.json")) {
    auto index = read_json(directory / "model.safetensors.index.json");
    for (auto& [key, value] : index.at("weight_map").items())
      if (key.starts_with("decoder.")) files.insert(value.get<std::string>());
  } else files.insert("model.safetensors");
  std::map<std::string, TensorSource> result;
  for (const auto& name : files) {
    fs::path relative(name);
    if (relative.is_absolute() || std::find(relative.begin(), relative.end(), "..") != relative.end())
      throw Error("ValueError", "Invalid VAE shard path");
    auto path = directory / relative;
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw Error("FileNotFoundError", "Missing VAE weights: " + path.string());
    uint64_t header_size = 0;
    stream.read(reinterpret_cast<char*>(&header_size), 8);
    auto file_size = fs::file_size(path);
    if (!stream || file_size < 8 || header_size > file_size - 8 || header_size > 100000000)
      throw Error("ValueError", "Invalid safetensors header: " + path.string());
    std::string header(header_size, '\0');
    stream.read(header.data(), header.size());
    if (!stream) throw Error("ValueError", "Truncated VAE header");
    auto metadata = Json::parse(header);
    for (auto& [key, value] : metadata.items()) {
      if (!key.starts_with("decoder.")) continue;
      if (value.at("dtype") != "F32") throw Error("ValueError", "VAE export contains tensors that are not FP32");
      auto dims = value.at("shape").get<std::vector<int64_t>>();
      uint64_t count = 1;
      for (auto dim : dims) {
        if (dim < 1 || uint64_t(dim) > std::numeric_limits<uint64_t>::max() / count)
          throw Error("ValueError", "Invalid VAE tensor shape");
        count *= dim;
      }
      auto offsets = value.at("data_offsets").get<std::vector<uint64_t>>();
      if (offsets.size() != 2 || offsets[0] > offsets[1] || offsets[1] > file_size - header_size - 8 ||
          count > std::numeric_limits<uint64_t>::max() / 4 || offsets[1] - offsets[0] != count * 4)
        throw Error("ValueError", "Invalid VAE tensor offsets");
      if (!result.emplace(key, TensorSource{path, 8 + header_size + offsets[0], count * 4, std::move(dims)}).second)
        throw Error("ValueError", "Duplicate VAE tensor: " + key);
    }
  }
  return result;
}
struct GraphRun {
  MPSGraph* graph = [MPSGraph new];
  MPSGraphTensor* input = nil;
  MPSGraphTensor* output = nil;
  NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [NSMutableDictionary new];
  MPSGraphTensor* parameter(const Tensor& t) {
    auto p = [graph placeholderWithShape:shape(t.dims) dataType:MPSDataTypeFloat32 name:nil];
    feeds[p] = t.data;
    return p;
  }
  MPSGraphTensor* scalar(float value) { return [graph constantWithScalar:value dataType:MPSDataTypeFloat32]; }
  MPSGraphTensor* add(MPSGraphTensor* a, MPSGraphTensor* b) { return [graph additionWithPrimaryTensor:a secondaryTensor:b name:nil]; }
  MPSGraphTensor* mul(MPSGraphTensor* a, MPSGraphTensor* b) { return [graph multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil]; }
  MPSGraphTensor* div(MPSGraphTensor* a, MPSGraphTensor* b) { return [graph divisionWithPrimaryTensor:a secondaryTensor:b name:nil]; }
  MPSGraphTensorData* run(id<MTLCommandQueue> queue, MPSGraphTensorData* value = nil) {
    uint64_t output_bytes = sizeof(float);
    for (NSNumber* extent in output.shape) {
      const auto dimension = extent.unsignedLongLongValue;
      if (!dimension || output_bytes > UINT64_MAX / dimension)
        throw Error("ValueError", "Invalid decoder graph output size");
      output_bytes *= dimension;
    }
    check_metal_allocation(output_bytes);
    if (input) feeds[input] = value;
    auto compilation = [MPSGraphCompilationDescriptor new];
    compilation.optimizationLevel = MPSGraphOptimizationLevel0;
    compilation.reducedPrecisionFastMath = MPSGraphReducedPrecisionFastMathNone;
    auto execution = [MPSGraphExecutionDescriptor new];
    execution.waitUntilCompleted = YES;
    execution.compilationDescriptor = compilation;
    __block NSError* failure = nil;
    execution.completionHandler = ^(MPSGraphTensorDataDictionary*, NSError* error) { failure = error; };
    MPSGraphTensorDataDictionary* results;
    @try {
      results = [graph runAsyncWithMTLCommandQueue:queue feeds:feeds targetTensors:@[output]
        targetOperations:nil executionDescriptor:execution];
    } @catch (NSException* exception) {
      if (input) [feeds removeObjectForKey:input];
      const char* reason = exception.reason.UTF8String;
      throw Error("RuntimeError", reason ? reason : "MPSGraph VAE exception");
    }
    if (failure) {
      if (input) [feeds removeObjectForKey:input];
      throw Error("RuntimeError", std::string(failure.localizedDescription.UTF8String));
    }
    if (input) [feeds removeObjectForKey:input];
    if (!results[output]) throw Error("RuntimeError", "MPSGraph VAE execution returned no output");
    return results[output];
  }
};
struct Conv { Tensor weight, bias; int in, out, kernel, stride, padding, dilation; bool transpose; };
struct Activation { Tensor alpha, beta; bool snake; };
struct Residual { Activation first, second; Conv conv, point; };
struct Block {
  Activation activation;
  Conv up;
  std::vector<Residual> residuals;
  // At most three tile shapes retained; feeds share device weights.
  std::map<int64_t, std::unique_ptr<GraphRun>> graphs;
};
MPSGraphTensor* activate(GraphRun& r, MPSGraphTensor* x, const Activation& a) {
  if (!a.snake) {
    auto positive = [r.graph maximumWithPrimaryTensor:x secondaryTensor:r.scalar(0) name:nil];
    auto negative = [r.graph minimumWithPrimaryTensor:x secondaryTensor:r.scalar(0) name:nil];
    auto em1 = [r.graph subtractionWithPrimaryTensor:[r.graph exponentWithTensor:negative name:nil] secondaryTensor:r.scalar(1) name:nil];
    return r.add(positive, em1);
  }
  auto alpha = [r.graph exponentWithTensor:r.parameter(a.alpha) name:nil];
  auto beta = [r.graph exponentWithTensor:r.parameter(a.beta) name:nil];
  auto sinusoid = [r.graph sinWithTensor:r.mul(x, alpha) name:nil];
  auto square = [r.graph squareWithTensor:sinusoid name:nil];
  return r.add(x, r.mul(r.div(r.scalar(1), r.add(beta, r.scalar(1e-9f))), square));
}
MPSGraphTensor* convolve(GraphRun& r, MPSGraphTensor* x, const Conv& c, int64_t& length) {
  auto descriptor = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:c.stride strideInY:1
    dilationRateInX:c.dilation dilationRateInY:1 groups:1 paddingLeft:c.padding paddingRight:c.padding
    paddingTop:0 paddingBottom:0 paddingStyle:MPSGraphPaddingStyleExplicit
    dataLayout:MPSGraphTensorNamedDataLayoutNCHW weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
  MPSGraphTensor* output;
  if (c.transpose) {
    length = (length - 1) * c.stride - 2 * c.padding + c.dilation * (c.kernel - 1) + 1;
    output = [r.graph convolutionTranspose2DWithSourceTensor:x weightsTensor:r.parameter(c.weight)
      outputShape:@[@1, @(c.out), @1, @(length)] descriptor:descriptor name:nil];
  } else {
    length = (length + 2*c.padding - c.dilation*(c.kernel-1) - 1) / c.stride + 1;
    output = [r.graph convolution2DWithSourceTensor:x weightsTensor:r.parameter(c.weight) descriptor:descriptor name:nil];
  }
  if (c.bias.data) output = r.add(output, r.parameter(c.bias));
  return output;
}
int64_t floor_div(int64_t a, int64_t b) { auto q = a/b; return q - (a%b < 0); }
}

struct VAE::Impl {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue = [device newCommandQueue];
  std::mutex mutex;
  Cancelled loading_guard;
  Conv first, last;
  Activation final_activation;
  bool final_tanh;
  int latent_dim, channels, ratio;
  std::vector<Block> blocks;
  std::map<int64_t, std::unique_ptr<GraphRun>> first_graphs, last_graphs;
  std::map<std::string, TensorSource> sources;

  Tensor load(const std::string& name, const std::vector<int64_t>& expected, const std::vector<int64_t>& reshaped) {
    check_cancel(loading_guard);
    auto it = sources.find(name);
    if (it == sources.end()) throw Error("ValueError", "Missing VAE tensor: " + name);
    const auto& source = it->second;
    if (source.dims != expected) throw Error("ValueError", "VAE tensor shape mismatch: " + name);
    check_metal_allocation(source.bytes);
    auto buffer = [device newBufferWithLength:source.bytes options:MTLResourceStorageModeShared];
    if (!buffer) throw Error("MemoryError", "Unable to allocate VAE weight buffer");
    std::ifstream input(source.file, std::ios::binary);
    input.seekg(source.offset);
    input.read(static_cast<char*>(buffer.contents), source.bytes);
    if (!input) throw Error("ValueError", "Truncated VAE tensor: " + name);
    const float* data = static_cast<const float*>(buffer.contents);
    for (uint64_t i = 0; i < source.bytes / 4; ++i)
      if (!std::isfinite(data[i])) throw Error("ValueError", "Non-finite VAE tensor: " + name);
    Tensor result{reshaped, [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer shape:shape(reshaped) dataType:MPSDataTypeFloat32]};
    sources.erase(it);
    return result;
  }
  Conv conv(const std::string& prefix, int in, int out, int kernel, int stride=1, int padding=0, int dilation=1, bool transpose=false, bool bias=true) {
    @autoreleasepool {
    const int a = transpose ? in : out, b = transpose ? out : in;
    auto v = load(prefix + ".weight_v", {a,b,kernel}, {a,b,kernel});
    auto g = load(prefix + ".weight_g", {a,1,1}, {a,1,1});
    // torch.nn.utils.weight_norm uses dim=0 for both convolution variants.
    // PyTorch 2.10 WeightNorm.mm uses rank-3 sum-of-squares, then (v/norm)*g.
    GraphRun norm;
    auto vp = norm.parameter(v);
    auto sum = [norm.graph reductionSumWithTensor:[norm.graph squareWithTensor:vp name:nil] axes:@[@1,@2] name:nil];
    auto normalized = norm.mul(norm.div(vp, [norm.graph squareRootWithTensor:sum name:nil]), norm.parameter(g));
    norm.output = [norm.graph reshapeTensor:normalized withShape:@[@(a),@(b),@1,@(kernel)] name:nil];
    check_cancel(loading_guard);
    Tensor weight{{a,b,1,kernel}, norm.run(queue)};
    Tensor offset;
    if (bias) offset = load(prefix + ".bias", {out}, {1,out,1,1});
    return {std::move(weight), std::move(offset), in,out,kernel,stride,padding,dilation,transpose};
    }
  }
  Activation activation(const std::string& prefix, int count, bool snake) {
    if (!snake) return {{},{},false};
    return {load(prefix + ".alpha", {count}, {1,count,1,1}), load(prefix + ".beta", {count}, {1,count,1,1}), true};
  }
  explicit Impl(const fs::path& directory, const Cancelled& cancelled) : loading_guard(cancelled) {
    if (!device || !queue) throw Error("RuntimeError", "Metal device unavailable for VAE");
    auto config = read_json(directory / "config.json");
    auto decoder = config.value("decoder_config", Json{{"out_channels",2},{"channels",64},{"c_mults",{1,2,4,8,16,32}},
      {"strides",{2,2,4,4,5,6}},{"latent_dim",64},{"use_snake",true},{"snake_type","vanilla"},{"final_tanh",false}});
    latent_dim = decoder.value("latent_dim",32);
    channels = decoder.value("out_channels",2);
    ratio = config.value("downsampling_ratio",1920);
    if (latent_dim != 64 || channels != 2 || config.value("sample_rate",48000) != 48000 ||
        config.value("latent_dim",64) != latent_dim || config.value("audio_channels",2) != channels)
      throw Error("ValueError", "VAE must decode 64-channel latents to 48kHz stereo");
    for (const auto* key : {"antialias_activation","use_nearest_upsample","use_filter"})
      if (decoder.value(key,false)) throw Error("ValueError", "Unsupported option for the released decoder");
    bool snake = decoder.value("use_snake",false);
    if (snake && decoder.value("snake_type",std::string("vanilla")) != "vanilla")
      throw Error("ValueError", "The released decoder uses vanilla SnakeBeta");
    final_tanh = decoder.value("final_tanh",true);
    auto mults = decoder.value("c_mults",std::vector<int>{1,2,4,8});
    auto strides = decoder.value("strides",std::vector<int>{2,4,8,8});
    int base = decoder.value("channels",128);
    if (mults.empty() || mults.size() != strides.size() || base < 1 || base > 8192)
      throw Error("ValueError", "Invalid VAE decoder dimensions");
    int64_t product = 1, length = 1;
    for (size_t i = 0; i < strides.size(); ++i) {
      if (strides[i] < 1 || strides[i] > 64 || mults[i] < 1 || mults[i] > 8192/base)
        throw Error("ValueError", "Invalid VAE decoder stride/channels");
      product *= strides[i];
      if (product > 1920) throw Error("ValueError", "Decoder strides do not match downsampling_ratio");
    }
    for (auto it = strides.rbegin(); it != strides.rend(); ++it) length = (length-1)*(*it) - 2*((*it+1)/2) + 2*(*it);
    if (ratio != 1920 || product != ratio || length != 1920-64)
      throw Error("ValueError", "VAE decoder must have natural output length 1920*T-64");
    sources = tensor_sources(directory);
    mults.insert(mults.begin(),1);
    first = conv("decoder.layers.0", latent_dim, base*mults.back(), 7,1,3);
    for (int i = static_cast<int>(mults.size())-1, stage = 1; i > 0; --i, ++stage) {
      @autoreleasepool {
        int in = base*mults[i], out = base*mults[i-1], stride = strides[i-1];
        auto prefix = "decoder.layers." + std::to_string(stage) + ".layers.";
        Block block;
        block.activation = activation(prefix+"0",in,snake);
        block.up = conv(prefix+"1",in,out,2*stride,stride,(stride+1)/2,1,true);
        for (int r = 0; r < 3; ++r) {
          int dilation = r == 0 ? 1 : r == 1 ? 3 : 9;
          auto p = prefix + std::to_string(r+2) + ".layers.";
          block.residuals.push_back({activation(p+"0",out,snake), activation(p+"2",out,snake),
            conv(p+"1",out,out,7,1,3*dilation,dilation), conv(p+"3",out,out,1)});
        }
        blocks.push_back(std::move(block));
      }
    }
    auto n = std::to_string(mults.size());
    final_activation = activation("decoder.layers."+n,base,snake);
    last = conv("decoder.layers."+std::to_string(mults.size()+1),base,channels,7,1,3,1,false,false);
    if (!sources.empty()) throw Error("ValueError", "Unexpected VAE tensor: " + sources.begin()->first);
    loading_guard = {};
  }
  int64_t required_halo(int core) const {
    int64_t low = 0, high = int64_t(core)*ratio-1;
    low -= 3; high += 3; // final conv
    for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
      for (auto r = it->residuals.rbegin(); r != it->residuals.rend(); ++r) {
        low -= 3*r->conv.dilation; high += 3*r->conv.dilation;
      }
      const auto& c = it->up;
      low = -floor_div(-(low+c.padding-c.dilation*(c.kernel-1)),c.stride);
      high = floor_div(high+c.padding,c.stride);
    }
    low -= 3; high += 3;
    return std::max({int64_t(0),-low,high-core+1});
  }
  GraphRun& graph(std::map<int64_t,std::unique_ptr<GraphRun>>& cache, int64_t length, int in,
                  const std::function<MPSGraphTensor*(GraphRun&,MPSGraphTensor*,int64_t&)>& build) {
    auto it = cache.find(length);
    if (it != cache.end()) return *it->second;
    if (cache.size() >= 3) cache.erase(cache.begin());
    auto run = std::make_unique<GraphRun>();
    run->input = [run->graph placeholderWithShape:@[@1,@(in),@1,@(length)] dataType:MPSDataTypeFloat32 name:nil];
    auto output_length = length;
    run->output = build(*run,run->input,output_length);
    return *cache.emplace(length,std::move(run)).first->second;
  }
  MPSGraphTensorData* tile(const FloatMatrix& latent, int64_t left, int64_t right, const Cancelled& cancelled) {
    int64_t length = right-left;
    check_cancel(cancelled);
    check_metal_allocation(length*latent_dim*sizeof(float));
    auto buffer = [device newBufferWithLength:length*latent_dim*sizeof(float) options:MTLResourceStorageModeShared];
    if (!buffer) throw Error("MemoryError", "Unable to allocate VAE latent tile");
    float* data = static_cast<float*>(buffer.contents);
    for (int c = 0; c < latent_dim; ++c)
      for (int64_t t = 0; t < length; ++t) data[c*length+t] = latent.values[(left+t)*latent_dim+c];
    MPSGraphTensorData* value = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer shape:@[@1,@(latent_dim),@1,@(length)] dataType:MPSDataTypeFloat32];
    check_cancel(cancelled);
    value = graph(first_graphs,length,latent_dim,[&](GraphRun& r,MPSGraphTensor* x,int64_t& n){ return convolve(r,x,first,n); }).run(queue,value);
    for (auto& block : blocks) {
      @autoreleasepool {
        check_cancel(cancelled);
        auto& run = graph(block.graphs,length,block.up.in,[&](GraphRun& r,MPSGraphTensor* x,int64_t& n) {
          x = convolve(r,activate(r,x,block.activation),block.up,n);
          for (const auto& residual : block.residuals) {
            auto skip = x;
            x = convolve(r,activate(r,x,residual.first),residual.conv,n);
            x = convolve(r,activate(r,x,residual.second),residual.point,n);
            x = r.add(x,skip);
          }
          return x;
        });
        value = run.run(queue,value);
        length = (length-1)*block.up.stride - 2*block.up.padding + block.up.kernel;
      }
    }
    check_cancel(cancelled);
    return graph(last_graphs,length,last.in,[&](GraphRun& r,MPSGraphTensor* x,int64_t& n) {
      x = convolve(r,activate(r,x,final_activation),last,n);
      return final_tanh ? [r.graph tanhWithTensor:x name:nil] : x;
    }).run(queue,value);
  }
};
VAE::VAE(const fs::path& directory, const Cancelled& cancelled) {
  @autoreleasepool { impl_ = std::make_unique<Impl>(directory,cancelled); }
}
VAE::~VAE() = default;
FloatMatrix VAE::decode(const FloatMatrix& latent, int core_frames, int halo_frames,
                        const Cancelled& cancelled, const StepCallback& on_progress) {
  latent.validate("VAE latents",64);
  if (core_frames < 1) throw Error("ValueError", "core_frames must be a positive integer");
  auto required = impl_->required_halo(core_frames);
  if (halo_frames < required) throw Error("ValueError", "halo_frames must be at least " + std::to_string(required) + " for this decoder");
  if (latent.rows > (std::numeric_limits<int64_t>::max()-64)/1920)
    throw Error("ValueError", "VAE audio length overflow");
  std::lock_guard lock(impl_->mutex);
  check_cancel(cancelled);
  const int64_t samples = latent.rows*1920-64;
  FloatMatrix audio{samples,2,{}};
  audio.values.resize(static_cast<size_t>(samples)*2);
  const int64_t tiles = (latent.rows+core_frames-1)/core_frames;
  if (tiles > std::numeric_limits<int>::max()) throw Error("ValueError", "Too many VAE tiles");
  int completed = 0;
  for (int64_t start = 0; start < latent.rows; start += core_frames) {
    @autoreleasepool {
      check_cancel(cancelled);
      const int64_t end = std::min(latent.rows,start+core_frames);
      const int64_t left = std::max(int64_t(0),start-halo_frames), right = std::min(latent.rows,end+halo_frames);
      auto tile = impl_->tile(latent,left,right,cancelled);
      const int64_t tile_length = (right-left)*1920-64;
      const int64_t out_start = start*1920, out_end = std::min(end*1920,samples), crop_start = (start-left)*1920;
      if (crop_start+out_end-out_start > tile_length)
        throw Error("RuntimeError", "VAE tile did not cover its requested output core");
      std::vector<float> decoded(static_cast<size_t>(tile_length)*2);
      [tile.mpsndarray readBytes:decoded.data() strideBytes:nullptr];
      for (float sample : decoded) if (!std::isfinite(sample)) throw Error("ValueError", "VAE audio contains non-finite values");
      for (int64_t i = 0; i < out_end-out_start; ++i) {
        audio.values[(out_start+i)*2] = std::clamp(decoded[crop_start+i],-1.0f,1.0f);
        audio.values[(out_start+i)*2+1] = std::clamp(decoded[tile_length+crop_start+i],-1.0f,1.0f);
      }
      check_cancel(cancelled);
      if (on_progress) on_progress(++completed,static_cast<int>(tiles));
    }
  }
  return audio;
}
}
