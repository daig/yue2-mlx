#pragma once
#include "types.hpp"
#include <memory>
namespace lyra {
class Tokenizer {
public:
  explicit Tokenizer(const fs::path &vocabulary);
  ~Tokenizer();
  Tokenizer(Tokenizer &&) noexcept;
  Tokenizer &operator=(Tokenizer &&) noexcept;
  std::vector<int> encode(std::string_view text) const;
  std::string decode(const std::vector<int> &tokens) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
std::string_view instruction(std::string_view cot);
std::vector<int>
token_prefixes(const SongRequest &, const Tokenizer &,
               const std::optional<std::vector<int>> &abc_ids = std::nullopt);
std::vector<int>
negative_prefix(const SongRequest &, const Tokenizer &,
                const std::optional<std::vector<int>> &abc_ids = std::nullopt);
std::vector<std::pair<int, int>> chunk_ranges(int frames, int prefix_tokens,
                                              int context = CONTEXT);
Json request_kwargs(Json data, const fs::path &base = fs::current_path());
} // namespace lyra
