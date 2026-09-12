#pragma once
#include "types.hpp"
#include <span>
namespace lyra {
std::string read_text(const fs::path &);
Json read_json(const fs::path &);
void write_json(const fs::path &, const Json &);
void write_text(const fs::path &, std::string_view);
void write_buffers(const fs::path &, std::span<const std::string_view>);
std::string canonical_json(const Json &);
std::string sha256_file(const fs::path &);
std::string identity(const Json &);
Json file_record(const fs::path &);
Json collect_hashes(const fs::path &,
                    const std::vector<std::string> &exclude = {"result.json"});
Json model_identity(const fs::path &, bool verify = true);
Json verify_result(const fs::path &,
                   const std::optional<std::string> &expected = std::nullopt);
fs::path expand_user(const fs::path &);
std::string unique_suffix();
std::string exception_type(const std::exception &);
} // namespace lyra
