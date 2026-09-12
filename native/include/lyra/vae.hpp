#pragma once
#include "types.hpp"
#include <memory>
namespace lyra {
class VAE {
 public:
  explicit VAE(const fs::path& directory,const Cancelled& cancelled={});
  ~VAE();
  FloatMatrix decode(const FloatMatrix&,int core_frames=256,int halo_frames=16,
   const Cancelled& cancelled={},const StepCallback& on_progress={});
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
FloatMatrix initial_noise(int frames,uint64_t seed);
void write_audio(const fs::path&,const FloatMatrix&,int sample_rate=48000);
struct AudioInfo { int64_t frames; int channels; int sample_rate; std::string format; };
AudioInfo audio_info(const fs::path&);
}
