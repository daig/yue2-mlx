#include "lyra/conversion.hpp"
#include "lyra/runtime.hpp"
#include <mlx/mlx.h>
#include <curl/curl.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <sys/file.h>
#include <unistd.h>
#include <stdio.h>

namespace lyra {
namespace {
namespace mx = mlx::core;
using Shape = std::vector<int>;
struct Spec { std::string dtype; Shape shape; uint64_t start=0,end=0; };
using Specs = std::map<std::string,Spec>;
constexpr size_t BUFFER=8*1024*1024;
[[noreturn]] void invalid(const std::string& message) { throw Error("ValueError",message); }
void require(bool condition,const std::string& message) { if(!condition) invalid(message); }
void os_error(const std::string& message) { throw Error("OSError",message+": "+std::strerror(errno)); }
struct FD {
 int value=-1;
 explicit FD(int descriptor):value(descriptor) { if(value<0) os_error("Opening file"); }
 ~FD() { if(value>=0) ::close(value); }
 FD(const FD&)=delete;
 FD& operator=(const FD&)=delete;
};
void sync_directory(const fs::path& path) { FD fd(::open(path.c_str(),O_RDONLY|O_DIRECTORY)); if(::fsync(fd.value)) os_error("Synchronizing directory"); }
void sync_file(const fs::path& path) { FD fd(::open(path.c_str(),O_RDONLY|O_NOFOLLOW)); if(::fsync(fd.value)) os_error("Synchronizing file"); }
struct Lock {
 FD fd;
 explicit Lock(const fs::path& path):fd(::open(path.c_str(),O_CREAT|O_RDWR|O_NOFOLLOW,0600)) {
  while(::flock(fd.value,LOCK_EX)) { if(errno!=EINTR) os_error("Locking conversion"); check_cancelled(); }
 }
 ~Lock() { ::flock(fd.value,LOCK_UN); }
};
struct Stage {
 fs::path path;
 explicit Stage(const fs::path& parent,const std::string& prefix) {
  std::string pattern=(parent/(prefix+".XXXXXX")).string();
  if(!::mkdtemp(pattern.data())) os_error("Creating conversion staging directory");
  path=pattern;
 }
 ~Stage() { std::error_code error; fs::remove_all(path,error); }
};
Json generator_files() {
 return {
 {"config.json",{{"bytes",959},{"sha256","ad3477bbef890bf98ae196c1e4b44779494a6231c4ab66f32708eabadf265329"}}},
 {"qwen.tiktoken",{{"bytes",2561218},{"sha256","b2b1b8dfb5cc5f024bafc373121c6aba3f66f9a5a0269e243470a1de16a33186"}}},
 {"weights_manifest.json",{{"bytes",179},{"sha256","2296f82c29dcb11aeefde07e216403a32fc86c8f6da2ec15a70102abfd936a93"}}},
 {"model.safetensors",{{"bytes",7261441640ULL},{"sha256","1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9"}}},
 {"LICENSE",{{"bytes",20309},{"sha256","060985741d20e70613b4c189c7de106cabd3fb2109fbbfb6d705c9d619417dd0"}}},
 {"THIRD_PARTY_NOTICES.md",{{"bytes",793},{"sha256","14d3fd9f6fee86b4260b69b0979735b99ffec8c6b4db254567047a4215cd9af3"}}},
 {"licenses/SnakeBeta-NVIDIA-MIT.txt",{{"bytes",1076},{"sha256","da9858d516047d82096d01c112a61bd67f26d289039464d668a1d45f91738ecc"}}},
 {"licenses/stable-audio-tools-MIT.txt",{{"bytes",1069},{"sha256","a1fac33b7bcd791b74fb33aeb439f825e7277e239fc119fb7d2ab6f084a0c101"}}}};
}
const std::vector<std::string> COPY={"config.json","qwen.tiktoken","LICENSE","THIRD_PARTY_NOTICES.md","licenses/SnakeBeta-NVIDIA-MIT.txt","licenses/stable-audio-tools-MIT.txt"};
std::set<std::string> keys(const Json& value) { require(value.is_object(),"Expected JSON object"); std::set<std::string> result; for(auto it=value.begin();it!=value.end();++it) result.insert(it.key()); return result; }
void exact(const Json& value,std::initializer_list<std::string> expected,const std::string& description) { require(keys(value)==std::set<std::string>(expected),"Invalid "+description); }
bool same(const Json& a,const Json& b) {
 if(a.is_number_integer() && b.is_number_integer()) return a==b;
 if(a.type()!=b.type()) return false;
 if(a.is_object()) { if(keys(a)!=keys(b)) return false; for(auto it=a.begin();it!=a.end();++it) if(!same(it.value(),b.at(it.key()))) return false; return true; }
 if(a.is_array()) { if(a.size()!=b.size()) return false; for(size_t i=0;i<a.size();++i) if(!same(a[i],b[i])) return false; return true; }
 return a==b;
}
Json decode(const std::string& bytes,const std::string& description) {
 std::vector<std::set<std::string>> objects;
 try {
  return Json::parse(bytes,[&](int,Json::parse_event_t event,Json& parsed) {
   if(event==Json::parse_event_t::object_start) objects.emplace_back();
   else if(event==Json::parse_event_t::key) require(objects.back().insert(parsed.get<std::string>()).second,"Duplicate JSON key in "+description);
   else if(event==Json::parse_event_t::object_end) objects.pop_back();
   return true;
  });
 } catch(const Json::exception& error) { invalid("Invalid "+description+": "+error.what()); }
}
Json bounded_json(const fs::path& path) {
 if(!fs::is_regular_file(path)) throw Error("FileNotFoundError",path.string());
 require(fs::file_size(path)<=16*1024*1024,"JSON file is unexpectedly large: "+path.string());
 return decode(read_text(path),path.filename().string());
}
void safe_name(const std::string& name) {
 require(!name.empty() && name.find('\\')==std::string::npos && name.find('\0')==std::string::npos,"Unsafe filename");
 fs::path p(name); require(!p.is_absolute() && p.generic_string()==name,"Unsafe filename: "+name);
 for(const auto& part:p) require(part!="." && part!=".." && !part.empty(),"Unsafe filename: "+name);
}
Json expected_config() { return {{"model_type","yue2"},{"architectures",{"YuE2ForCausalLM"}},{"hidden_size",2048},{"num_hidden_layers",28},{"num_attention_heads",16},{"num_key_value_heads",8},{"head_dim",128},{"intermediate_size",6144},{"vocab_size",184704},{"rms_norm_eps",1e-6},{"rope_theta",1000000},{"max_position_embeddings",24576},{"tie_word_embeddings",false},{"latent_type","vae"},{"latent_dim",64},{"max_latent_frames",24576},{"timestep_shift",1.0}}; }
void validate_config(const Json& config) {
 require(config.is_object(),"Generator config must be a JSON object");
 auto expected=expected_config(); for(auto it=expected.begin();it!=expected.end();++it) require(config.contains(it.key()) && same(config.at(it.key()),it.value()),"Pinned generator config mismatch: "+it.key());
 require(config.value("dtype",config.value("torch_dtype",std::string()))=="bfloat16","Pinned generator must declare bfloat16 weights");
}
Specs expected_specs() {
 Specs result;
 auto add=[&](const std::string& name,Shape shape) { result.emplace(name,Spec{"BF16",std::move(shape)}); };
 add("model.embed_tokens.weight",{184704,2048}); add("model.norm.weight",{2048}); add("lm_head.weight",{184704,2048});
 add("llm2vae.weight",{64,2048}); add("llm2vae.bias",{64}); add("vae2llm.weight",{2048,64}); add("vae2llm.bias",{2048});
 add("time_embedder.mlp.0.weight",{2048,256}); add("time_embedder.mlp.0.bias",{2048}); add("time_embedder.mlp.2.weight",{2048,2048}); add("time_embedder.mlp.2.bias",{2048}); add("latent_pos_embed.pe",{24576,2048});
 for(int i=0;i<28;++i) {
  auto p="model.layers."+std::to_string(i)+".";
  for(auto n:{"input_layernorm","post_attention_layernorm","nar_input_layernorm","nar_pre_mlp_layernorm"}) add(p+n+".weight",{2048});
  for(auto a:{"self_attn.","nar_self_attn."}) {
   for(auto n:{"q_proj","o_proj"}) add(p+a+n+".weight",{2048,2048});
   for(auto n:{"k_proj","v_proj"}) add(p+a+n+".weight",{1024,2048});
   for(auto n:{"q_norm","k_norm"}) add(p+a+n+".weight",{128});
  }
  for(auto a:{"mlp.","nar_mlp."}) { for(auto n:{"gate_proj","up_proj"}) add(p+a+n+".weight",{6144,2048}); add(p+a+"down_proj.weight",{2048,6144}); }
 }
 return result;
}
bool ar_key(const std::string& name) { return name=="model.embed_tokens.weight" || name=="model.norm.weight" || name=="lm_head.weight" || (name.starts_with("model.layers.") && name.find(".nar_")==std::string::npos); }
bool linear_key(const std::string& name) { return name=="lm_head.weight" || (ar_key(name) && (name.find("_proj.weight")!=std::string::npos)); }
Specs partition_specs(bool ar) { Specs result; for(auto& [name,spec]:expected_specs()) if(ar_key(name)==ar) result.emplace(name,spec); return result; }
Specs quantized_specs(int bits) {
 auto result=partition_specs(true);
 for(auto& [name,spec]:partition_specs(true)) if(linear_key(name)) {
  result.at(name)={"U32",{spec.shape[0],spec.shape[1]*bits/32}};
  std::string prefix=name.substr(0,name.size()-6);
  result.emplace(prefix+"scales",Spec{"BF16",{spec.shape[0],spec.shape[1]/64}});
  result.emplace(prefix+"biases",Spec{"BF16",{spec.shape[0],spec.shape[1]/64}});
 }
 return result;
}
uint64_t tensor_bytes(const Spec& spec) {
 uint64_t count=spec.dtype=="BF16"?2:4;
 for(int d:spec.shape) { require(d>0 && count<=std::numeric_limits<uint64_t>::max()/uint64_t(d),"Tensor shape overflow"); count*=d; }
 return count;
}
struct Header { Specs specs; Json metadata; uint64_t data; };
Header read_header(const fs::path& path) {
 uint64_t size=fs::file_size(path),length=0; require(size>=10,"Invalid safetensors file: "+path.string());
 std::ifstream in(path,std::ios::binary); in.read(reinterpret_cast<char*>(&length),8);
 require(in.good() && length>=2 && length<=64*1024*1024 && length<=size-8,"Invalid safetensors header size");
 std::string bytes(length,'\0'); in.read(bytes.data(),length); require(in.good(),"Truncated safetensors header");
 Json raw=decode(bytes,"safetensors header"); require(raw.is_object(),"Invalid safetensors header");
 Header result{{},raw.value("__metadata__",Json::object()),length+8};
 if(result.metadata.is_null()) result.metadata=Json::object();
 require(result.metadata.is_object(),"Invalid safetensors metadata"); for(auto& v:result.metadata) require(v.is_string(),"Invalid safetensors metadata value");
 raw.erase("__metadata__"); std::vector<std::pair<uint64_t,uint64_t>> ranges;
 for(auto it=raw.begin();it!=raw.end();++it) {
  require(!it.key().empty(),"Empty tensor name"); const auto& v=it.value(); exact(v,{"dtype","shape","data_offsets"},"tensor entry");
  require(v["dtype"]=="BF16" || v["dtype"]=="U32","Unsupported tensor dtype");
  require(v["shape"].is_array() && !v["shape"].empty(),"Invalid tensor shape");
  Shape shape; for(const auto& d:v["shape"]) { require(d.is_number_integer() && d>0 && d<=std::numeric_limits<int>::max(),"Invalid tensor dimension"); shape.push_back(d.get<int>()); }
  const auto& offsets=v["data_offsets"]; require(offsets.is_array() && offsets.size()==2,"Invalid tensor offsets");
  for(auto& o:offsets) require(o.is_number_integer() && o>=0,"Invalid tensor offset");
  Spec spec{v["dtype"].get<std::string>(),std::move(shape),offsets[0].get<uint64_t>(),offsets[1].get<uint64_t>()};
  require(spec.end>=spec.start && spec.end-spec.start==tensor_bytes(spec),"Tensor byte size mismatch: "+it.key());
  ranges.emplace_back(spec.start,spec.end); result.specs.emplace(it.key(),std::move(spec));
 }
 std::sort(ranges.begin(),ranges.end()); uint64_t position=0;
 for(auto [start,end]:ranges) { require(start==position,"Non-contiguous or overlapping tensor data"); position=end; }
 require(position==size-result.data,"Safetensors payload length mismatch"); return result;
}
void validate_specs(const Specs& actual,const Specs& expected) {
 require(actual.size()==expected.size(),"Tensor set mismatch");
 for(auto& [name,spec]:expected) { auto it=actual.find(name); require(it!=actual.end(),"Missing tensor: "+name); require(it->second.shape==spec.shape,"Tensor shape mismatch: "+name); require(it->second.dtype==spec.dtype,"Tensor dtype mismatch: "+name); }
}
Json metadata(const std::string& partition,const std::string& precision) {
 Json value={{"format","mlx"},{"partition",partition},{"precision",precision}};
 if(precision!="bfloat16") value["quantization"]="affine-group64-linear";
 return value;
}
std::vector<std::string> ordered(const Specs& specs) {
 std::vector<std::string> names; for(auto& [name,spec]:specs) names.push_back(name);
 std::sort(names.begin(),names.end(),[&](const auto& a,const auto& b) { return specs.at(a).start<specs.at(b).start; }); return names;
}
void write_all(int fd,const void* data,size_t length) {
 auto p=static_cast<const char*>(data);
 while(length) { check_cancelled(); ssize_t n=::write(fd,p,std::min(length,BUFFER)); if(n<0) { if(errno==EINTR) continue; os_error("Writing converted tensor"); } require(n>0,"Short write"); p+=n; length-=n; }
}
void write_header(int fd,const Specs& specs,const std::vector<std::string>& names,const Json& meta) {
 Json header={{"__metadata__",meta}}; uint64_t offset=0;
 for(const auto& name:names) { const auto& spec=specs.at(name); uint64_t next=offset+tensor_bytes(spec); header[name]={{"dtype",spec.dtype},{"shape",spec.shape},{"data_offsets",{offset,next}}}; offset=next; }
 std::string text=header.dump(); text.append((8-text.size()%8)%8,' '); uint64_t length=text.size(); write_all(fd,&length,8); write_all(fd,text.data(),text.size());
}
void copy_range(std::ifstream& source,int destination,uint64_t offset,uint64_t length,std::vector<char>& buffer) {
 source.seekg(offset); require(source.good(),"Cannot seek source tensor");
 while(length) { check_cancelled(); size_t count=std::min<uint64_t>(length,buffer.size()); source.read(buffer.data(),count); require(size_t(source.gcount())==count,"Source tensor ended unexpectedly"); write_all(destination,buffer.data(),count); length-=count; }
}
Json write_partition(const fs::path& source,const fs::path& destination,bool ar) {
 Header h=read_header(source); auto expected=partition_specs(ar); std::vector<std::string> names;
 for(const auto& name:ordered(h.specs)) if(expected.contains(name)) names.push_back(name);
 require(names.size()==expected.size(),"Missing partition tensors");
 FD out(::open(destination.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600)); write_header(out.value,expected,names,metadata(ar?"ar":"nar","bfloat16"));
 std::ifstream in(source,std::ios::binary); std::vector<char> buffer(BUFFER);
 for(auto& name:names) { const auto& spec=h.specs.at(name); copy_range(in,out.value,h.data+spec.start,spec.end-spec.start,buffer); }
 if(::fsync(out.value)) os_error("Synchronizing partition"); return file_record(destination);
}
Json write_quantized(const fs::path& source,const fs::path& destination,int bits) {
 Header h=read_header(source); validate_specs(h.specs,partition_specs(true)); auto expected=quantized_specs(bits); auto names=ordered(h.specs); std::vector<std::string> outputs;
 for(auto& name:names) { outputs.push_back(name); if(linear_key(name)) { auto p=name.substr(0,name.size()-6); outputs.push_back(p+"scales"); outputs.push_back(p+"biases"); } }
 GPUExecution guard; FD out(::open(destination.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600));
 write_header(out.value,expected,outputs,metadata("ar",std::to_string(bits)+"bit"));
 auto weights=mx::load_safetensors(source.string()).first;
 require(weights.size()==h.specs.size(),"MLX loaded a different AR tensor set");
 std::ifstream raw(source,std::ios::binary); std::vector<char> buffer(BUFFER);
 for(const auto& name:names) {
  guard.check(); const auto& spec=h.specs.at(name);
  if(!linear_key(name)) copy_range(raw,out.value,h.data+spec.start,spec.end-spec.start,buffer);
  else {
   auto arrays=mx::quantize(weights.at(name),64,bits,"affine",std::nullopt,mx::Device(mx::Device::gpu)); mx::eval(arrays);
   require(arrays.size()==3,"MLX quantization output count mismatch"); auto p=name.substr(0,name.size()-6); std::array<std::string,3> outnames={name,p+"scales",p+"biases"};
   for(size_t i=0;i<3;++i) {
    auto& array=arrays[i]; const auto& e=expected.at(outnames[i]);
    require(std::equal(array.shape().begin(),array.shape().end(),e.shape.begin(),e.shape.end()) && array.dtype()==(i==0?mx::uint32:mx::bfloat16),"MLX emitted incompatible quantization data: "+name);
    if(i==0) write_all(out.value,array.data<uint32_t>(),array.nbytes());
    else write_all(out.value,array.data<mx::bfloat16_t>(),array.nbytes());
   }
  }
  weights.erase(name); mx::clear_cache(); guard.check();
 }
 require(weights.empty(),"Not every AR tensor was consumed"); if(::fsync(out.value)) os_error("Synchronizing quantized tensors");
 auto result=read_header(destination); validate_specs(result.specs,expected); require(result.metadata==metadata("ar",std::to_string(bits)+"bit"),"Quantized metadata mismatch"); return file_record(destination);
}
Json settings(bool native) {
 return {{"generator_dtype","bfloat16"},{"ar_extraction","yue2.fast.ar_keys"},{"nar_partition","all source tensors not selected by ar_keys"},{"positional_buffer_dtype","bfloat16"},{"quantization",{{"mode","affine"},{"group_size",64},{"modules","mlx.nn.Linear"},{"linear_only",true},{"embedding_dtype","bfloat16"}}},{"versions",native?Json{{"mlx","0.32.2"},{"mlx_lm","native-ar-extraction-v1"},{"safetensors","native-safetensors-v1"}}:Json{{"mlx","0.32.2"},{"mlx_lm","0.31.3"},{"safetensors","0.7.0"}}}};
}
Json precision_record(const std::string& precision) {
 if(precision=="bf16") return {{"ar","ar-bf16.safetensors"},{"nar","nar-bf16.safetensors"},{"dtype","bfloat16"}};
 require(precision=="8bit" || precision=="4bit","precision must be one of: bf16, 8bit, 4bit");
 return {{"ar","ar-"+precision+".safetensors"},{"bits",precision=="8bit"?8:4},{"group_size",64},{"mode","affine"},{"linear_only",true},{"embedding_dtype","bfloat16"}};
}
Json provenance() { return {{"repository",MODEL_REPO},{"revision",MODEL_REVISION},{"revision_proof","verified-pinned-content"},{"identity",{{"files",generator_files()}}}}; }
Json upstream() { return {{"repository","https://github.com/multimodal-art-projection/YuE"},{"commit",UPSTREAM_COMMIT}}; }
void verify_pinned(const fs::path& directory,const Json& expected,bool hashes) {
 if(!fs::is_directory(directory)) throw Error("FileNotFoundError",directory.string());
 std::set<std::string> weights;
 for(auto& entry:fs::recursive_directory_iterator(directory)) if(entry.is_regular_file() && entry.path().extension()==".safetensors") weights.insert(entry.path().lexically_relative(directory).generic_string());
 require(weights==std::set<std::string>{"model.safetensors"},"Pinned checkpoint needs exactly model.safetensors");
 for(auto it=expected.begin();it!=expected.end();++it) {
  check_cancelled(); auto file=directory/it.key(); if(!fs::is_regular_file(file)) throw Error("FileNotFoundError","Pinned checkpoint file is missing: "+it.key());
  require(fs::file_size(file)==it.value()["bytes"].get<uint64_t>(),"Pinned checkpoint size mismatch: "+it.key());
  if(hashes || it.key()!="model.safetensors") require(sha256_file(file)==it.value()["sha256"].get_ref<const std::string&>(),"Pinned checkpoint content mismatch: "+it.key());
 }
 auto manifest=bounded_json(directory/"weights_manifest.json"); exact(manifest,{"schema","files"},"source weight manifest");
 require(same(manifest,Json{{"schema",1},{"files",{{"model.safetensors",expected.at("model.safetensors")}}}}),"Source weight manifest does not identify the pinned checkpoint");
}
Json validate_conversion(const fs::path& directory,const Json& manifest,bool hashes) {
 exact(manifest,{"schema","format","source","upstream","conversion","precisions","files"},"conversion manifest");
 require(same(manifest["schema"],Json(1)) && manifest["format"]=="lyra-yue2-mlx","Unsupported conversion schema or format");
 require(same(manifest["source"],provenance()),"Converted source provenance is not the pinned generator");
 require(same(manifest["upstream"],upstream()),"Incompatible upstream implementation");
 require(same(manifest["conversion"],settings(false)) || same(manifest["conversion"],settings(true)),"Incompatible conversion settings");
 auto& precisions=manifest["precisions"]; require(precisions.is_object() && precisions.contains("bf16"),"Invalid precision manifest");
 std::set<std::string> expected(COPY.begin(),COPY.end()); expected.insert("ar-bf16.safetensors"); expected.insert("nar-bf16.safetensors");
 for(auto it=precisions.begin();it!=precisions.end();++it) { require(same(it.value(),precision_record(it.key())),"Incompatible precision settings"); expected.insert("ar-"+it.key()+".safetensors"); }
 const auto& files=manifest["files"]; require(keys(files)==expected,"Converted file manifest mismatch");
 std::set<std::string> actual,dirs;
 for(auto& entry:fs::recursive_directory_iterator(directory)) {
  auto relative=entry.path().lexically_relative(directory).generic_string(); auto status=entry.symlink_status();
  require(!fs::is_symlink(status),"Converted artifacts cannot contain symlinks: "+relative);
  if(fs::is_regular_file(status)) actual.insert(relative); else if(fs::is_directory(status)) dirs.insert(relative); else invalid("Unsupported converted artifact: "+relative);
 }
 auto all=expected; all.insert("conversion.json"); require(actual==all && dirs==std::set<std::string>{"licenses"},"Converted directory has missing or unexpected entries");
 static const std::regex digest("[0-9a-f]{64}");
 for(auto it=files.begin();it!=files.end();++it) {
  safe_name(it.key()); exact(it.value(),{"bytes","sha256"},"file record"); auto& r=it.value();
  require(r["bytes"].is_number_integer() && r["bytes"]>0 && r["sha256"].is_string() && std::regex_match(r["sha256"].get<std::string>(),digest),"Invalid file record: "+it.key());
  require(fs::file_size(directory/it.key())==r["bytes"].get<uint64_t>(),"Converted file size mismatch: "+it.key());
  if(hashes) { check_cancelled(); require(sha256_file(directory/it.key())==r["sha256"].get_ref<const std::string&>(),"Converted file integrity failed: "+it.key()); }
 }
 auto source=generator_files(); for(auto& name:COPY) require(same(files.at(name),source.at(name)),"Copied pinned file identity changed: "+name);
 validate_config(bounded_json(directory/"config.json"));
 for(bool ar:{true,false}) { auto h=read_header(directory/(ar?"ar-bf16.safetensors":"nar-bf16.safetensors")); validate_specs(h.specs,partition_specs(ar)); require(h.metadata==metadata(ar?"ar":"nar","bfloat16"),"BF16 safetensors metadata incompatible"); }
 for(auto precision:{"8bit","4bit"}) if(precisions.contains(precision)) { auto h=read_header(directory/(std::string("ar-")+precision+".safetensors")); validate_specs(h.specs,quantized_specs(precision[0]-'0')); require(h.metadata==metadata("ar",precision),"Quantized safetensors metadata incompatible"); }
 return manifest;
}
std::string environment(const char* key) { auto p=std::getenv(key); return p?p:""; }
bool enabled(const char* key) { auto v=environment(key); std::transform(v.begin(),v.end(),v.begin(),[](unsigned char c){ return std::toupper(c); }); return v=="1" || v=="ON" || v=="YES" || v=="TRUE"; }
fs::path cache_root(const std::optional<fs::path>& cache) {
 if(cache) return fs::absolute(expand_user(*cache));
 for(auto key:{"HF_HUB_CACHE","HUGGINGFACE_HUB_CACHE"}) if(auto v=environment(key);!v.empty()) return fs::absolute(expand_user(v));
 if(auto home=environment("HF_HOME");!home.empty()) return fs::absolute(expand_user(home))/"hub";
 auto xdg=environment("XDG_CACHE_HOME"); return (xdg.empty()?expand_user("~/.cache"):expand_user(xdg))/"huggingface/hub";
}
struct Curl {
 CURL* handle=nullptr;
 Curl() { static const int initialized=[] { if(curl_global_init(CURL_GLOBAL_DEFAULT)!=CURLE_OK) throw Error("RuntimeError","Initializing libcurl failed"); return 1; }(); (void)initialized; handle=curl_easy_init(); if(!handle) throw Error("MemoryError","Allocating HTTP client failed"); }
 ~Curl() { curl_easy_cleanup(handle); }
};
struct Transfer { int fd=-1; std::string text; bool failed=false; uint64_t bytes=0,limit=0; };
size_t receive(char* ptr,size_t size,size_t count,void* opaque) {
 auto& transfer=*static_cast<Transfer*>(opaque); size_t n=size*count;
 if(transfer.limit && (transfer.bytes>transfer.limit || n>transfer.limit-transfer.bytes)) { transfer.failed=true; return 0; }
 try { if(transfer.fd>=0) write_all(transfer.fd,ptr,n); else transfer.text.append(ptr,n); transfer.bytes+=n; return n; } catch(...) { transfer.failed=true; return 0; }
}
int progress(void*,curl_off_t,curl_off_t,curl_off_t,curl_off_t) { return cancellation_requested()?1:0; }
std::string token() {
 if(enabled("HF_HUB_DISABLE_IMPLICIT_TOKEN")) return {};
 auto value=environment("HF_TOKEN"); if(value.empty()) value=environment("HUGGING_FACE_HUB_TOKEN"); if(!value.empty()) return value;
 auto explicit_path=environment("HF_TOKEN_PATH"); auto home=environment("HF_HOME");
 auto path=explicit_path.empty()?(home.empty()?cache_root(std::nullopt).parent_path():expand_user(home))/"token":expand_user(explicit_path);
 if(fs::is_regular_file(path)) { value=read_text(path); while(!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back(); }
 return value;
}
std::string escape_path(const std::string& path) {
 Curl curl; std::string result; size_t start=0;
 while(start<=path.size()) { auto end=path.find('/',start); auto part=path.substr(start,end==std::string::npos?std::string::npos:end-start); char* escaped=curl_easy_escape(curl.handle,part.data(),int(part.size())); if(!escaped) throw Error("MemoryError","Encoding Hub URL failed"); result+=escaped; curl_free(escaped); if(end==std::string::npos) break; result+='/'; start=end+1; }
 return result;
}
void request(const std::string& url,Transfer& transfer) {
 Curl curl; char error[CURL_ERROR_SIZE]={}; auto bearer=token(); curl_slist* raw_headers=nullptr;
 if(!bearer.empty()) raw_headers=curl_slist_append(raw_headers,("Authorization: Bearer "+bearer).c_str());
 std::unique_ptr<curl_slist,decltype(&curl_slist_free_all)> headers(raw_headers,curl_slist_free_all);
 curl_easy_setopt(curl.handle,CURLOPT_URL,url.c_str()); curl_easy_setopt(curl.handle,CURLOPT_FOLLOWLOCATION,1L); curl_easy_setopt(curl.handle,CURLOPT_MAXREDIRS,10L);
 curl_easy_setopt(curl.handle,CURLOPT_PROTOCOLS_STR,"https"); curl_easy_setopt(curl.handle,CURLOPT_REDIR_PROTOCOLS_STR,"https");
 curl_easy_setopt(curl.handle,CURLOPT_FAILONERROR,1L); curl_easy_setopt(curl.handle,CURLOPT_HTTPHEADER,headers.get()); curl_easy_setopt(curl.handle,CURLOPT_USERAGENT,"lyra-native/1");
 curl_easy_setopt(curl.handle,CURLOPT_CONNECTTIMEOUT,30L); curl_easy_setopt(curl.handle,CURLOPT_LOW_SPEED_LIMIT,1L); curl_easy_setopt(curl.handle,CURLOPT_LOW_SPEED_TIME,60L);
 curl_easy_setopt(curl.handle,CURLOPT_WRITEFUNCTION,receive); curl_easy_setopt(curl.handle,CURLOPT_WRITEDATA,&transfer); curl_easy_setopt(curl.handle,CURLOPT_ERRORBUFFER,error);
 curl_easy_setopt(curl.handle,CURLOPT_NOPROGRESS,0L); curl_easy_setopt(curl.handle,CURLOPT_XFERINFOFUNCTION,progress);
 auto status=curl_easy_perform(curl.handle); check_cancelled();
 if(status!=CURLE_OK) throw Error("OSError","Hub acquisition failed: "+url+": "+(error[0]?std::string(error):curl_easy_strerror(status)));
 require(!transfer.failed,"Hub response exceeded file limit or could not be written");
}
std::string endpoint() { auto value=environment("HF_ENDPOINT"); if(value.empty()) value="https://huggingface.co"; while(value.ends_with('/')) value.pop_back(); require(value.starts_with("https://"),"HF_ENDPOINT must use HTTPS"); return value; }
bool allowed(const std::string& name) {
 static const std::set<std::string> files={"config.json","generation_config.json","yue2_generation_config.json","weights_manifest.json","model.safetensors","model.safetensors.index.json","qwen.tiktoken","modeling_yue2.py","modeling_vae.py","LICENSE","THIRD_PARTY_NOTICES.md","licenses/stable-audio-tools-MIT.txt","licenses/SnakeBeta-NVIDIA-MIT.txt"};
 static const std::regex shard("model-[0-9]{5}-of-[0-9]{5}\\.safetensors"); return files.contains(name) || std::regex_match(name,shard);
}
fs::path acquire(const std::string& repo,const std::string& requested,bool offline,const std::optional<fs::path>& cache,const Json& pinned) {
 safe_name(repo); require(std::count(repo.begin(),repo.end(),'/')<=1,"Hub model must be a repository or owner/repository identifier"); safe_name(requested);
 std::string encoded=repo; size_t slash=encoded.find('/'); if(slash!=std::string::npos) encoded.replace(slash,1,"--"); auto root=cache_root(cache); auto repository=root/("models--"+encoded);
 offline=offline || enabled("HF_HUB_OFFLINE"); std::string revision=requested;
 static const std::regex commit("[0-9a-f]{40}");
 if(!std::regex_match(revision,commit) && fs::is_regular_file(repository/"refs"/revision)) { revision=read_text(repository/"refs"/revision); while(!revision.empty() && std::isspace(static_cast<unsigned char>(revision.back()))) revision.pop_back(); }
 Json files=pinned;
 auto snapshot=repository/"snapshots"/revision;
 if(offline) {
  if(!std::regex_match(revision,commit) || !fs::is_directory(snapshot)) throw Error("FileNotFoundError","Pinned Hub snapshot is unavailable in offline cache: "+repo+"@"+requested);
  if(!pinned.empty()) verify_pinned(snapshot,pinned,false);
  else { bool found=false; for(auto& entry:fs::recursive_directory_iterator(snapshot)) if(entry.is_regular_file() && allowed(entry.path().lexically_relative(snapshot).generic_string())) found=true; require(found,"Cached Hub snapshot contains no model files"); }
  return fs::canonical(snapshot);
 }
 if(!pinned.empty() && fs::is_directory(snapshot)) {
  bool complete=true; for(auto it=pinned.begin();it!=pinned.end();++it) if(!fs::is_regular_file(snapshot/it.key()) || fs::file_size(snapshot/it.key())!=it.value()["bytes"].get<uint64_t>()) { complete=false; break; }
  if(complete) { verify_pinned(snapshot,pinned,false); return fs::canonical(snapshot); }
 }
 if(pinned.empty()) {
  Transfer metadata_transfer; metadata_transfer.limit=16*1024*1024;
  request(endpoint()+"/api/models/"+escape_path(repo)+"/revision/"+escape_path(requested)+"?blobs=true",metadata_transfer);
  auto info=decode(metadata_transfer.text,"Hub model metadata"); require(info.is_object() && info.contains("sha") && info["sha"].is_string(),"Hub metadata has no commit"); revision=info["sha"].get<std::string>(); require(std::regex_match(revision,commit),"Invalid Hub commit");
  require(info.contains("siblings") && info["siblings"].is_array(),"Hub metadata has no file list"); files=Json::object();
  for(auto& item:info["siblings"]) { auto name=item.at("rfilename").get<std::string>(); if(!allowed(name)) continue; safe_name(name); Json record=Json::object(); if(item.contains("size")) record["bytes"]=item["size"]; if(item.contains("lfs") && item["lfs"].is_object()) { const auto& lfs=item["lfs"]; if(lfs.contains("sha256")) record["sha256"]=lfs["sha256"]; if(lfs.contains("size")) record["bytes"]=lfs["size"]; } files[name]=record; }
  require(!files.empty(),"Hub repository contains no supported model files"); snapshot=repository/"snapshots"/revision;
 }
 fs::create_directories(snapshot); auto locks=root/".locks"/("models--"+encoded); fs::create_directories(locks);
 Lock lock(locks/(revision+".lyra.lock"));
 for(auto it=files.begin();it!=files.end();++it) {
  check_cancelled(); safe_name(it.key()); const auto& record=it.value(); auto target=snapshot/it.key();
  if(fs::is_regular_file(target)) {
   require(!record.contains("bytes") || fs::file_size(target)==record["bytes"].get<uint64_t>(),"Cached Hub file size mismatch: "+it.key());
   if(record.contains("sha256") && it.key()!="model.safetensors") require(sha256_file(target)==record["sha256"].get_ref<const std::string&>(),"Cached Hub file content mismatch: "+it.key());
   continue;
  }
  fs::create_directories(target.parent_path()); Stage stage(repository,".lyra-download"); auto temporary=stage.path/"blob";
  { FD fd(::open(temporary.c_str(),O_CREAT|O_EXCL|O_WRONLY,0600)); Transfer transfer; transfer.fd=fd.value; if(record.contains("bytes")) transfer.limit=record["bytes"].get<uint64_t>();
   request(endpoint()+"/"+escape_path(repo)+"/resolve/"+revision+"/"+escape_path(it.key()),transfer);
   require(!record.contains("bytes") || transfer.bytes==record["bytes"].get<uint64_t>(),"Downloaded Hub file size mismatch: "+it.key()); if(::fsync(fd.value)) os_error("Synchronizing downloaded model"); }
  if(record.contains("sha256")) require(sha256_file(temporary)==record["sha256"].get_ref<const std::string&>(),"Downloaded Hub content mismatch: "+it.key());
  // Existing Hugging Face snapshots can contain blob symlinks. New native files
  // are installed as ordinary snapshot files; the standard cache layout remains usable.
  if(::renamex_np(temporary.c_str(),target.c_str(),RENAME_EXCL)) os_error("Installing Hub file"); sync_directory(target.parent_path());
 }
 if(!std::regex_match(requested,commit)) { auto ref=repository/"refs"/requested; fs::create_directories(ref.parent_path()); write_text(ref,revision); sync_file(ref); sync_directory(ref.parent_path()); }
 if(!pinned.empty()) verify_pinned(snapshot,pinned,false);
 return fs::canonical(snapshot);
}
} // namespace
Json pinned_vae_files() {
 auto files=generator_files(); files.erase("qwen.tiktoken");
 files["config.json"]={{"bytes",1378},{"sha256","f0191bb9694009956de44e0c361a6f1334760be4c8f848e599bde242a54a0970"}};
 files["weights_manifest.json"]={{"bytes",178},{"sha256","017d64a4d288217a43fcac3866a0e1d55f796fa891d4dfb6de0bd170ab750026"}};
 files["model.safetensors"]={{"bytes",530512720},{"sha256","807ce9d5149fa27c5ad3e6582058469852e908f6c5acc8c8aa338e7ab7751346"}}; return files;
}
Json verify_conversion(const fs::path& raw,bool verify_hashes) {
 auto path=expand_user(raw);
 if(fs::is_symlink(path) || !fs::is_directory(path)) throw Error("FileNotFoundError",path.string());
 require(!fs::is_symlink(path/"conversion.json"),"conversion.json cannot be a symlink");
 return validate_conversion(fs::canonical(path),bounded_json(path/"conversion.json"),verify_hashes);
}
fs::path resolve_model(const std::string& model,std::string_view revision,bool offline,const std::optional<fs::path>& cache_dir) {
 auto path=expand_user(model); if(fs::is_directory(path)) return fs::canonical(path);
 if(path.is_absolute() || model.starts_with('.')) throw Error("FileNotFoundError",path.string());
 std::string rev=revision.empty()?"main":std::string(revision); Json pinned=Json::object();
 if(model==MODEL_REPO && rev==MODEL_REVISION) pinned=generator_files(); else if(model==VAE_REPO && rev==VAE_REVISION) pinned=pinned_vae_files();
 return acquire(model,rev,offline,cache_dir,pinned);
}
std::pair<fs::path,fs::path> fetch_models(const std::optional<fs::path>& cache_dir,bool offline) {
 auto generator=acquire(std::string(MODEL_REPO),std::string(MODEL_REVISION),offline,cache_dir,generator_files());
 auto vae=acquire(std::string(VAE_REPO),std::string(VAE_REVISION),offline,cache_dir,pinned_vae_files()); return {generator,vae};
}
fs::path prepare(const fs::path& raw_source,const fs::path& raw_output,std::string_view requested_precision) {
 std::string precision(requested_precision); precision_record(precision);
 auto source=expand_user(raw_source); if(!fs::is_directory(source)) throw Error("FileNotFoundError",source.string()); source=fs::canonical(source);
 auto output=expand_user(raw_output); require(!fs::is_symlink(output),"Converted output directory cannot be a symlink"); output=fs::weakly_canonical(fs::absolute(output));
 auto relative=output.lexically_relative(source); require(output!=source && (relative.empty() || *relative.begin()==".."),"Converted output must be outside the source checkpoint");
 fs::create_directories(output.parent_path()); Lock lock(output.parent_path()/("."+output.filename().string()+".conversion.lock"));
 verify_pinned(source,generator_files(),true); validate_config(bounded_json(source/"config.json")); validate_specs(read_header(source/"model.safetensors").specs,expected_specs());
 bool existing=fs::exists(output); Json manifest;
 if(existing) {
  if(fs::is_symlink(output) || !fs::is_directory(output)) throw Error("FileExistsError","Conversion destination is not a model directory");
  manifest=verify_conversion(output,true); if(manifest["precisions"].contains(precision)) return output;
 }
 Stage stage(output.parent_path(),"."+output.filename().string()+(existing?".incremental":".fresh"));
 if(existing) {
  for(auto it=manifest["files"].begin();it!=manifest["files"].end();++it) { auto target=stage.path/it.key(); fs::create_directories(target.parent_path()); if(::linkat(AT_FDCWD,(output/it.key()).c_str(),AT_FDCWD,target.c_str(),0)) os_error("Linking existing conversion"); }
  // Incremental conversion is native even when its BF16 partition was produced by Python.
  manifest["conversion"]=settings(true);
 } else {
  Json records=Json::object();
  for(auto& name:COPY) { auto target=stage.path/name; fs::create_directories(target.parent_path()); fs::copy_file(source/name,target,fs::copy_options::none); sync_file(target); records[name]=file_record(target); require(same(records[name],generator_files().at(name)),"Pinned file changed while copying: "+name); }
  records["ar-bf16.safetensors"]=write_partition(source/"model.safetensors",stage.path/"ar-bf16.safetensors",true);
  records["nar-bf16.safetensors"]=write_partition(source/"model.safetensors",stage.path/"nar-bf16.safetensors",false);
  manifest={{"schema",1},{"format","lyra-yue2-mlx"},{"source",provenance()},{"upstream",upstream()},{"conversion",settings(true)},{"precisions",{{"bf16",precision_record("bf16")}}},{"files",records}};
 }
 if(precision!="bf16") {
  auto name="ar-"+precision+".safetensors"; manifest["files"][name]=write_quantized(stage.path/"ar-bf16.safetensors",stage.path/name,precision=="8bit"?8:4); manifest["precisions"][precision]=precision_record(precision);
 }
 write_json(stage.path/"conversion.json",manifest); sync_file(stage.path/"conversion.json"); validate_conversion(stage.path,manifest,false);
 sync_directory(stage.path/"licenses"); sync_directory(stage.path);
 if(::renamex_np(stage.path.c_str(),output.c_str(),existing?RENAME_SWAP:RENAME_EXCL)) os_error("Atomically installing converted model");
 sync_directory(output.parent_path()); return output;
}
} // namespace lyra
