/*
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou,
Iain Melvin, Jason Weston) Copyright (c) 2006      Idiap Research Institute
(Samy Bengio) Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert,
Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories
America and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

MT19937 upstream attribution:
A C-program for MT19937, with initialization improved 2002/2/10.
Coded by Takuji Nishimura and Makoto Matsumoto.
This is a faster version by taking Shawn Cokus's optimization,
Matthe Bellew's simplification, Isaku Wada's real version.
 *
Before using, initialize the state by using init_genrand(seed)
or init_by_array(init_key, key_length).
 *
Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
All rights reserved.
 *
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 *
  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
 *
  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
 *
  3. The names of its contributors may not be used to endorse or promote
  products derived from this software without specific prior written
  permission.
 *
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "lyra/vae.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <numbers>

namespace lyra {
namespace {
// Adapted from PyTorch 2.10 ATen/core/MT19937RNGEngine.h and
// ATen/native/cpu/DistributionTemplates.h: arm64 non-AVX/VSX normal_fill.
class TorchMT19937 {
  std::array<uint32_t, 624> state_{};
  int left_ = 1;
  unsigned next_ = 0;
  static uint32_t twist(uint32_t u, uint32_t v) {
    return (((u & 0x80000000u) | (v & 0x7fffffffu)) >> 1) ^
           ((v & 1) ? 0x9908b0dfu : 0u);
  }
  void advance() {
    left_ = 624;
    next_ = 0;
    for (unsigned i = 0; i < 227; ++i)
      state_[i] = state_[i + 397] ^ twist(state_[i], state_[i + 1]);
    for (unsigned i = 227; i < 623; ++i)
      state_[i] = state_[i - 227] ^ twist(state_[i], state_[i + 1]);
    state_[623] = state_[396] ^ twist(state_[623], state_[0]);
  }

public:
  explicit TorchMT19937(uint64_t seed) {
    state_[0] = static_cast<uint32_t>(seed);
    for (uint32_t i = 1; i < 624; ++i)
      state_[i] = 1812433253u * (state_[i - 1] ^ (state_[i - 1] >> 30)) + i;
  }
  uint32_t next() {
    if (--left_ == 0)
      advance();
    uint32_t y = state_[next_++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
  }
  float uniform() { return static_cast<float>(next() & 0xffffffu) * 0x1p-24f; }
};
void normal_block(float *data) {
  for (int j = 0; j < 8; ++j) {
    const float u1 = 1.0f - data[j];
    const float u2 = data[j + 8];
    const float radius = std::sqrt(-2.0f * std::log(u1));
    // Torch deliberately multiplies by double pi, then rounds theta to FP32.
    const float theta = 2.0f * std::numbers::pi_v<double> * u2;
    data[j] = radius * std::cos(theta) * 1.0f + 0.0f;
    data[j + 8] = radius * std::sin(theta) * 1.0f + 0.0f;
  }
}
} // namespace
FloatMatrix initial_noise(int frames, uint64_t seed) {
  if (frames < 1)
    throw Error("ValueError", "frames must be positive");
  FloatMatrix result{frames, 64, {}};
  result.values.resize(static_cast<size_t>(frames) * 64);
  TorchMT19937 generator(seed);
  for (float &value : result.values)
    value = generator.uniform();
  const size_t size = result.values.size();
  for (size_t i = 0; i + 15 < size; i += 16)
    normal_block(result.values.data() + i);
  // The public [frames,64] contract is always divisible by 16. This is the
  // upstream tail rule, retained explicitly rather than changing draw order.
  if (size % 16) {
    float *tail = result.values.data() + size - 16;
    for (int i = 0; i < 16; ++i)
      tail[i] = generator.uniform();
    normal_block(tail);
  }
  return result;
}
} // namespace lyra
