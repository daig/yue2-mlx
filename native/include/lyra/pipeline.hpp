#pragma once
#include "ar.hpp"
#include "nar.hpp"
#include "protocol.hpp"
#include "runtime.hpp"
#include "vae.hpp"
namespace lyra {
struct PipelineOptions {
  std::string model="m-a-p/YuE2-3B",vae="m-a-p/YuE2-Vae",precision="bf16";
  fs::path converted_dir="models/converted";
  bool offline=false,progress=true,require_ac=false;
  double memory_budget_gib=DEFAULT_MEMORY_BUDGET_GIB;
  int vae_core_frames=256,query_chunk_size=256;
  GenerationConfig generation;
};
class Pipeline {
 public:
  explicit Pipeline(PipelineOptions);
  ~Pipeline();
  void close();
  SymbolicPlan plan(const SongRequest&,const Json& abc_sampling=nullptr);
  SemanticResult generate_semantic(const SymbolicPlan&,const Json& sampling=nullptr);
  FloatMatrix synthesize(const SemanticResult&,const FloatMatrix& noise);
  FloatMatrix decode(const FloatMatrix&);
  SongResult generate(const SongRequest&,const Json& abc_sampling=nullptr,const Json& semantic_sampling=nullptr);
  SongResult render(const SymbolicPlan&,const Json& semantic_sampling=nullptr);
  Json effective_config(const SongRequest&,const Json& abc_sampling=nullptr,const Json& semantic_sampling=nullptr) const;
  Json weights,load_timing=Json::object();
  const GenerationConfig& generation_config() const {return options_.generation;}
 private:
  PipelineOptions options_;
  fs::path model_dir_,vae_dir_;
  std::string runtime_sha256_;
  Json runtime_,decoder_release_;
  bool closed_=false;
  std::unique_ptr<GPUExecution> execution_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::shared_ptr<ARModel> ar_,bf16_ar_;
  std::unique_ptr<AcousticModel> nar_;
  std::unique_ptr<VAE> vae_;
  ARModel& load_ar();
  AcousticModel& load_nar();
  void record_load(const std::string&,double);
  SongResult render_with_config(const SymbolicPlan&,const Json&,Json,std::string,double);
  TokenGeneration generate_phase(const std::vector<int>&,const Sampling&,uint64_t,std::string_view,const std::vector<int>& negative={},double guidance=1,bool legacy_off=false);
};
}
