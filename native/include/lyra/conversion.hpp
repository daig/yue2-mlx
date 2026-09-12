#pragma once
#include "storage.hpp"
namespace lyra {
inline constexpr std::string_view MODEL_REPO="m-a-p/YuE2-3B";
inline constexpr std::string_view MODEL_REVISION="1a96eca688d6ae5d7f0feb88573fec89920fcd19";
inline constexpr std::string_view VAE_REPO="m-a-p/YuE2-Vae";
inline constexpr std::string_view VAE_REVISION="95535e72a97bc0f09b8ada125d26b4009428c0e8";
inline constexpr std::string_view UPSTREAM_COMMIT="92a73cc7652fcc1f937855e4b765e0a0edd7ff2e";
Json verify_conversion(const fs::path&,bool verify_hashes=true);
std::pair<fs::path,fs::path> fetch_models(const std::optional<fs::path>& cache_dir=std::nullopt,bool offline=false);
fs::path resolve_model(const std::string& model,std::string_view revision,bool offline=false,const std::optional<fs::path>& cache_dir=std::nullopt);
fs::path prepare(const fs::path& source,const fs::path& output,std::string_view precision="bf16");
Json pinned_vae_files();
}
