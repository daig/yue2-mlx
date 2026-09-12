#pragma once
#include "types.hpp"
#include <mlx/mlx.h>
#include <unordered_map>
namespace lyra {
namespace mx = mlx::core;
using Weights = std::unordered_map<std::string,mx::array>;
namespace model_ops {
mx::array linear(const mx::array&,const Weights&,const std::string& name,int bits=0,int group_size=64);
mx::array rms_norm(const mx::array&,const mx::array& weight,float eps);
mx::array silu(const mx::array&);
mx::array mlp(const mx::array&,const Weights&,const std::string& name,int bits=0);
mx::array rope(const mx::array&,double theta,int offset);
mx::array sdpa(const mx::array&,const mx::array&,const mx::array&,float scale,const std::optional<mx::array>& mask=std::nullopt);
}
struct KVCache {
  int capacity,offset=0;
  std::optional<mx::array> keys,values;
  explicit KVCache(int capacity);
  std::pair<mx::array,mx::array> update(const mx::array&,const mx::array&);
  void close();
};
class ARModel {
 public:
  Json config;
  Weights weights;
  std::string precision;
  ARModel(const fs::path&,std::string precision="bf16",bool verify=true);
  mx::array hidden(const mx::array& ids,std::vector<KVCache>& cache,const Cancelled& cancelled={});
  mx::array project(const mx::array& hidden,std::string_view phase) const;
  void materialize() const;
};
struct TokenGeneration { std::vector<int> tokens; Json timing; bool truncated=false; };
TokenGeneration generate_tokens(ARModel&,const std::vector<int>& prefix,const Sampling&,uint64_t seed,std::string_view phase,
 const std::vector<int>& negative={},double cfg_scale=1,bool legacy_off=false,
 const Cancelled& cancelled={},const TokenCallback& on_token={});
}
