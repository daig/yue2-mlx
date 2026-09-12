#pragma once
#include "ar.hpp"
namespace lyra {
class AcousticModel {
public:
  ARModel &ar;
  Json config;
  Weights weights;
  mx::array rope_inverse_frequency{0}, time_frequencies{0};
  mx::array timestep_shift{0}, timestep_shift_delta{0}, one{1.0f, mx::bfloat16};
  AcousticModel(const fs::path &, ARModel &, bool verify = true);
  void materialize() const;
};
FloatMatrix synthesize(AcousticModel &, const std::vector<int> &prefix,
                       const std::vector<int> &codec, const FloatMatrix &noise,
                       int steps = 32, int context = CONTEXT,
                       int query_chunk_size = 256,
                       const Cancelled &cancelled = {},
                       const StepCallback &on_progress = {});
bool mpp_attention_available();
} // namespace lyra
