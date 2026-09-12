#include "lyra/storage.hpp"
#include "lyra/runtime.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <set>
#include <unistd.h>

namespace lyra {
namespace {
[[noreturn]] void io_error(const fs::path& p) {
  const int e=errno;
  throw Error(e==ENOENT?"FileNotFoundError":e==EACCES||e==EPERM?"PermissionError":"OSError",p.string()+": "+std::strerror(e));
}
std::string hex_digest(const unsigned char* bytes) {
  constexpr char hex[]="0123456789abcdef";
  std::string result(CC_SHA256_DIGEST_LENGTH*2,'0');
  for(size_t i=0;i<CC_SHA256_DIGEST_LENGTH;++i) { result[2*i]=hex[bytes[i]>>4]; result[2*i+1]=hex[bytes[i]&15]; }
  return result;
}
void append_json(std::string& out,const Json& j) {
  if(j.is_number_float()) {
    double value=j.get<double>();
    if(!std::isfinite(value)) throw Error("ValueError","Out of range float values are not JSON compliant");
    if(value==0) { out+=std::signbit(value)?"-0.0":"0.0"; return; }
    // Shortest round-tripping significand, then CPython repr's fixed/scientific policy.
    std::array<char,64> buffer{};
    auto result=std::to_chars(buffer.data(),buffer.data()+buffer.size(),value,std::chars_format::scientific);
    if(result.ec!=std::errc{}) throw Error("ValueError","Cannot serialize floating-point value");
    std::string text(buffer.data(),result.ptr);
    size_t e=text.find('e');
    int exponent=std::stoi(text.substr(e+1));
    bool negative=text[0]=='-';
    std::string digits=text.substr(negative?1:0,e-(negative?1:0));
    digits.erase(std::remove(digits.begin(),digits.end(),'.'),digits.end());
    if(negative) out+='-';
    if(exponent < -4 || exponent >= 16) {
      out+=digits[0];
      if(digits.size()>1) { out+='.'; out+=digits.substr(1); }
      out+='e'; out+=exponent<0?'-':'+';
      auto power=std::to_string(std::abs(exponent));
      if(power.size()<2) out+='0';
      out+=power;
    } else {
      int point=exponent+1;
      if(point<=0) { out+="0."; out.append(-point,'0'); out+=digits; }
      else if(static_cast<size_t>(point)>=digits.size()) { out+=digits; out.append(point-digits.size(),'0'); out+=".0"; }
      else { out+=digits.substr(0,point); out+='.'; out+=digits.substr(point); }
    }
  } else if(j.is_array()) {
    out+='['; bool first=true;
    for(const auto& item:j) { if(!first) out+=','; first=false; append_json(out,item); } out+=']';
  } else if(j.is_object()) {
    out+='{'; bool first=true;
    // UTF-8 lexicographic ordering agrees with Unicode scalar ordering for valid UTF-8.
    for(auto it=j.begin();it!=j.end();++it) { if(!first) out+=','; first=false; out+=Json(it.key()).dump(-1,' ',false); out+=':'; append_json(out,it.value()); } out+='}';
  } else out+=j.dump(-1,' ',false);
}
fs::path safe_artifact(const fs::path& directory,const std::string& name) {
  fs::path relative(name);
  if(relative.empty()||relative.is_absolute()) throw Error("ValueError","Invalid artifact path");
  auto root=fs::canonical(directory);
  auto resolved=fs::weakly_canonical(root/relative);
  auto rel=resolved.lexically_relative(root);
  if(rel.empty()||*rel.begin()=="..") throw Error("ValueError","Invalid artifact path");
  return root/relative;
}
}
std::string read_text(const fs::path& path) {
  std::ifstream stream(path,std::ios::binary);
  if(!stream) io_error(path);
  std::string value((std::istreambuf_iterator<char>(stream)),{});
  if(stream.bad()) throw Error("OSError","Cannot read "+path.string());
  return value;
}
Json read_json(const fs::path& path) {
  try { return Json::parse(read_text(path)); }
  catch(const Json::exception& e) { throw Error("JSONDecodeError",e.what()); }
}
std::string canonical_json(const Json& value) { std::string out; append_json(out,value); return out; }
std::string identity(const Json& value) {
  auto text=canonical_json(value); CC_SHA256_CTX context; CC_SHA256_Init(&context);
  size_t offset=0;
  while(offset<text.size()) { auto count=std::min(text.size()-offset,size_t(std::numeric_limits<CC_LONG>::max())); CC_SHA256_Update(&context,text.data()+offset,static_cast<CC_LONG>(count)); offset+=count; }
  unsigned char digest[CC_SHA256_DIGEST_LENGTH]; CC_SHA256_Final(digest,&context); return hex_digest(digest);
}
std::string sha256_file(const fs::path& path) {
  std::ifstream stream(path,std::ios::binary); if(!stream) io_error(path);
  CC_SHA256_CTX context; CC_SHA256_Init(&context);
  std::vector<char> block(8*1024*1024);
  while(stream) { check_cancelled(); stream.read(block.data(),block.size()); auto n=stream.gcount(); if(n) CC_SHA256_Update(&context,block.data(),static_cast<CC_LONG>(n)); }
  if(!stream.eof()) throw Error("OSError","Cannot hash "+path.string());
  unsigned char digest[CC_SHA256_DIGEST_LENGTH]; CC_SHA256_Final(digest,&context); return hex_digest(digest);
}
std::string unique_suffix() {
  static std::atomic<uint64_t> sequence{0};
  return std::to_string(getpid())+"-"+std::to_string(sequence.fetch_add(1))+"-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
}
void write_buffers(const fs::path& path,std::span<const std::string_view> buffers) {
  if(!path.parent_path().empty()) fs::create_directories(path.parent_path());
  fs::path temporary=path.string()+"."+unique_suffix()+".tmp";
  int fd=open(temporary.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0666);
  if(fd<0) io_error(temporary);
  try {
    for(auto text:buffers) {
      size_t offset=0;
      while(offset<text.size()) { ssize_t n=::write(fd,text.data()+offset,std::min(text.size()-offset,size_t(1)<<30)); if(n<0&&errno==EINTR) continue; if(n<=0) io_error(temporary); offset+=n; }
    }
    if(fsync(fd)<0) io_error(temporary);
    int closing=fd; fd=-1; if(close(closing)<0) io_error(temporary);
    if(rename(temporary.c_str(),path.c_str())<0) io_error(path);
  } catch(...) { if(fd>=0) close(fd); std::error_code ec; fs::remove(temporary,ec); throw; }
}
void write_text(const fs::path& path,std::string_view text) { write_buffers(path,std::span(&text,1)); }
void write_json(const fs::path& path,const Json& value) {
  // Validate finite values before dump, which otherwise silently turns NaN into null.
  (void)canonical_json(value);
  write_text(path,value.dump(2,' ',false)+"\n");
}
Json file_record(const fs::path& path) { return {{"sha256",sha256_file(path)},{"bytes",fs::file_size(path)}}; }
Json collect_hashes(const fs::path& directory,const std::vector<std::string>& exclude) {
  Json result=Json::object();
  for(const auto& entry:fs::recursive_directory_iterator(directory)) {
    if(std::find(exclude.begin(),exclude.end(),entry.path().filename().string())!=exclude.end()) continue;
    auto name=entry.path().lexically_relative(directory).generic_string();
    auto path=safe_artifact(directory,name);
    if(fs::is_symlink(entry.symlink_status())&&fs::is_directory(path)) throw Error("ValueError","Invalid artifact directory symlink");
    if(fs::is_regular_file(path)) result[name]=file_record(path);
  }
  return result;
}
Json model_identity(const fs::path& directory,bool verify) {
  Json expected=nullptr, entries=Json::object();
  if(fs::exists(directory/"weights_manifest.json")) expected=read_json(directory/"weights_manifest.json");
  for(const auto& file:fs::directory_iterator(directory)) {
    if(file.path().extension()!=".safetensors") continue;
    auto name=file.path().filename().string(); auto record=file_record(file.path());
    if(verify&&!expected.is_null()&&(!expected.contains("files")||!expected["files"].contains(name)||expected["files"][name].value("sha256",Json())!=record["sha256"])) throw Error("ValueError","Weight integrity failed: "+name);
    entries[name]=std::move(record);
  }
  if(entries.empty()) throw Error("FileNotFoundError","No safetensors weights in "+directory.string());
  if(!expected.is_null()) {
    if(!expected.is_object()||!expected.contains("files")||!expected["files"].is_object()||expected["files"].size()!=entries.size()) throw Error("ValueError","Weight manifest has missing or unexpected shards");
    for(auto it=entries.begin();it!=entries.end();++it) if(!expected["files"].contains(it.key())) throw Error("ValueError","Weight manifest has missing or unexpected shards");
  }
  return {{"files",entries},{"config_sha256",sha256_file(directory/"config.json")}};
}
Json verify_result(const fs::path& directory,const std::optional<std::string>& expected) {
  auto result=read_json(safe_artifact(directory,"result.json"));
  if(!result.is_object()||result.value("status",Json())!="complete") throw Error("ValueError","Saved request did not complete");
  if(expected&&result.value("identity",Json())!=*expected) throw Error("ValueError","Request/config/weight identity changed; use a new output directory");
  auto artifacts=result.value("artifacts",Json::object());
  if(!artifacts.is_object()) throw Error("ValueError","Incomplete result artifact manifest");
  for(auto name:{"audio.flac","prefix.npy","semantic.npy","latent.npy","request.json","config.json"}) if(!artifacts.contains(name)) throw Error("ValueError","Incomplete result artifact manifest");
  for(auto it=artifacts.begin();it!=artifacts.end();++it) {
    auto p=safe_artifact(directory,it.key()); const auto& record=it.value();
    if(!record.is_object()||!record.contains("bytes")||!record["bytes"].is_number_integer()||!record.contains("sha256")||!record["sha256"].is_string()||!fs::is_regular_file(p)||Json(fs::file_size(p))!=record["bytes"]||sha256_file(p)!=record["sha256"].get_ref<const std::string&>()) throw Error("ValueError","Missing or corrupt result: "+it.key());
  }
  return result;
}
fs::path expand_user(const fs::path& path) {
  auto text=path.string();
  if(text=="~"||text.starts_with("~/")) { const char* home=getenv("HOME"); if(!home) throw Error("RuntimeError","Could not determine home directory"); return fs::path(home)/text.substr(text.size()==1?1:2); }
  return path;
}
std::string exception_type(const std::exception& e) {
  if(auto error=dynamic_cast<const Error*>(&e)) return error->type;
  if(dynamic_cast<const Json::parse_error*>(&e)) return "JSONDecodeError";
  if(dynamic_cast<const Json::type_error*>(&e)) return "TypeError";
  if(dynamic_cast<const Json::exception*>(&e)||dynamic_cast<const std::invalid_argument*>(&e)) return "ValueError";
  if(auto error=dynamic_cast<const fs::filesystem_error*>(&e)) { auto code=error->code(); if(code==std::errc::no_such_file_or_directory) return "FileNotFoundError"; if(code==std::errc::permission_denied) return "PermissionError"; if(code==std::errc::file_exists) return "FileExistsError"; return "OSError"; }
  if(dynamic_cast<const std::bad_alloc*>(&e)) return "MemoryError";
  return "RuntimeError";
}
}
