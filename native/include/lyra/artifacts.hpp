#pragma once
#include "types.hpp"
namespace lyra {
void save_plan(const SymbolicPlan &, const fs::path &);
SymbolicPlan load_plan(const fs::path &);
void save_plan_artifacts(const SymbolicPlan &, const fs::path &,
                         const GenerationConfig &);
std::pair<SymbolicPlan, std::optional<GenerationConfig>>
load_plan_artifacts(const fs::path &);
Json save_artifacts(const SongResult &, const fs::path &);
SavedSong load_artifacts(const fs::path &);
void save_ints(const fs::path &, const std::vector<int> &, bool int64 = false);
std::vector<int> load_ints(const fs::path &);
void save_matrix(const fs::path &, const FloatMatrix &);
FloatMatrix load_matrix(const fs::path &);
} // namespace lyra
