#include "lyra/vae.hpp"
#include <sndfile.h>
#include <algorithm>
#include <cctype>
#include <memory>

namespace lyra {
namespace {
using SoundFile = std::unique_ptr<SNDFILE, decltype(&sf_close)>;
std::string audio_error(const fs::path& path) {
  return "Audio file " + path.string() + ": " + sf_strerror(nullptr);
}
}
void write_audio(const fs::path& path, const FloatMatrix& audio, int sample_rate) {
  audio.validate("audio", 2);
  if (sample_rate <= 0) throw Error("ValueError", "sample_rate must be positive");
  auto extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return std::tolower(c); });
  SF_INFO info{};
  info.samplerate = sample_rate;
  info.channels = 2;
  if (extension == ".wav") info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
  else if (extension == ".flac") info.format = SF_FORMAT_FLAC | SF_FORMAT_PCM_24;
  else throw Error("ValueError", "Audio output must be WAV or FLAC");
  SoundFile file(sf_open(path.c_str(), SFM_WRITE, &info), sf_close);
  if (!file) throw Error("RuntimeError", audio_error(path));
  sf_command(file.get(), SFC_SET_CLIPPING, nullptr, SF_TRUE);
  if (sf_writef_float(file.get(), audio.values.data(), audio.rows) != audio.rows)
    throw Error("RuntimeError", "Failed to write complete audio: " + std::string(sf_strerror(file.get())));
  if (sf_close(file.release()) != 0) throw Error("RuntimeError", "Failed to close audio file: " + path.string());
}
AudioInfo audio_info(const fs::path& path) {
  SF_INFO info{};
  SoundFile file(sf_open(path.c_str(), SFM_READ, &info), sf_close);
  if (!file) throw Error("RuntimeError", audio_error(path));
  std::string format;
  switch (info.format & SF_FORMAT_TYPEMASK) {
    case SF_FORMAT_WAV: format = "WAV"; break;
    case SF_FORMAT_WAVEX: format = "WAVEX"; break;
    case SF_FORMAT_FLAC: format = "FLAC"; break;
    default: {
      SF_FORMAT_INFO detail{};
      detail.format = info.format & SF_FORMAT_TYPEMASK;
      if (sf_command(nullptr, SFC_GET_FORMAT_INFO, &detail, sizeof(detail)) == 0 && detail.name) format = detail.name;
      else throw Error("RuntimeError", "Unknown audio container format");
    }
  }
  return {info.frames, info.channels, info.samplerate, std::move(format)};
}
}
