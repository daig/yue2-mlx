#include "lyra/ar.hpp"
#include "lyra/conversion.hpp"
#include <cmath>
#include <set>
#include <map>

namespace lyra {
namespace {
mx::array rows(const mx::array& a,int start,int stop) {
  auto begin=mx::Shape(a.ndim(),0), end=a.shape(); begin[0]=start; end[0]=stop;
  return mx::slice(a,begin,end);
}
int precision_bits(std::string_view p) { return p=="bf16"?0:p=="8bit"?8:4; }
}
namespace model_ops {
mx::array linear(const mx::array& x,const Weights& w,const std::string& n,int bits,int group_size) {
  auto y=bits?mx::quantized_matmul(x,w.at(n+".weight"),w.at(n+".scales"),w.at(n+".biases"),true,group_size,bits,"affine"):
    mx::matmul(x,mx::swapaxes(w.at(n+".weight"),-1,-2));
  if(auto b=w.find(n+".bias");b!=w.end()) y=y+b->second;
  return y;
}
mx::array rms_norm(const mx::array& x,const mx::array& weight,float eps) {
  auto variance=mx::mean(mx::square(mx::astype(x,mx::float32)),-1,true);
  auto scale=mx::astype(mx::rsqrt(variance+mx::array(eps,mx::float32)),x.dtype());
  return mx::astype(mx::astype(x*scale,x.dtype())*weight,x.dtype());
}
mx::array silu(const mx::array& x) {
  static auto fused=mx::compile([](const std::vector<mx::array>& a) {
    auto v=mx::astype(a[0],mx::float32);
    return std::vector<mx::array>{mx::astype(v*mx::sigmoid(v),a[0].dtype())};
  },true);
  return fused({x})[0];
}
mx::array mlp(const mx::array& x,const Weights& w,const std::string& n,int bits) {
  auto gate=silu(linear(x,w,n+".gate_proj",bits));
  auto up=linear(x,w,n+".up_proj",bits);
  return linear(mx::astype(gate*up,x.dtype()),w,n+".down_proj",bits);
}
mx::array rope(const mx::array& x,double theta,int offset) {
  int d=x.shape(-1),length=x.shape(-2);
  if(d%2) throw Error("ValueError","RoPE head dimension must be even");
  // Derived frequencies retain FP32 arithmetic, independent of checkpoint dtype.
  static thread_local std::map<std::pair<int,double>,mx::array> frequencies;
  auto key=std::make_pair(d,theta);
  auto found=frequencies.find(key);
  if(found==frequencies.end()) {
    auto exponent=mx::arange(0,d,2,mx::float32)/mx::array(d,mx::float32);
    auto inverse=mx::array(1.0f)/mx::power(mx::array(float(theta),mx::float32),exponent);
    mx::eval(inverse);
    found=frequencies.emplace(key,std::move(inverse)).first;
  }
  const auto& inverse=found->second;
  auto positions=mx::arange(length,mx::float32)+mx::array(offset,mx::float32);
  auto angles=mx::expand_dims(positions,-1)*mx::expand_dims(inverse,0);
  auto cosine=mx::astype(mx::cos(angles),x.dtype()),sine=mx::astype(mx::sin(angles),x.dtype());
  auto begin=mx::Shape(x.ndim(),0),end=x.shape(); end.back()=d/2;
  auto first=mx::slice(x,begin,end); begin.back()=d/2;end.back()=d;
  auto second=mx::slice(x,begin,end);
  auto fc=mx::astype(first*cosine,x.dtype()),ss=mx::astype(second*sine,x.dtype());
  auto sc=mx::astype(second*cosine,x.dtype()),fs=mx::astype(first*sine,x.dtype());
  return mx::concatenate({mx::astype(fc-ss,x.dtype()),mx::astype(sc+fs,x.dtype())},-1);
}
mx::array sdpa(const mx::array& q,const mx::array& k,const mx::array& v,float scale,const std::optional<mx::array>& mask) {
  if(q.shape(2)<=8&&!mask) return mx::fast::scaled_dot_product_attention(q,k,v,scale);
  return mx::astype(mx::fast::scaled_dot_product_attention(mx::astype(q,mx::float32),mx::astype(k,mx::float32),mx::astype(v,mx::float32),scale,"",mask),q.dtype());
}
}
KVCache::KVCache(int size):capacity(size) {
  if(size<1||size>CONTEXT) throw Error("ValueError","KV cache capacity must be in [1, 24576]");
}
std::pair<mx::array,mx::array> KVCache::update(const mx::array& k,const mx::array& v) {
  if(k.ndim()!=4||k.shape()!=v.shape()||k.shape(0)!=1) throw Error("ValueError","KV updates must be matching [1, H, T, D] arrays");
  if(keys&&(keys->shape(1)!=k.shape(1)||keys->shape(3)!=k.shape(3))) throw Error("ValueError","KV update head shape does not match the cache");
  int end=offset+k.shape(2);
  if(end>capacity) throw Error("ValueError","KV cache capacity exceeded; generation was not shortened");
  if(!keys) { auto shape=k.shape();shape[2]=capacity;keys=mx::zeros(shape,mx::bfloat16);values=mx::zeros(shape,mx::bfloat16); }
  keys=mx::slice_update(*keys,mx::astype(k,mx::bfloat16),{0,0,offset,0},{1,k.shape(1),end,k.shape(3)});
  values=mx::slice_update(*values,mx::astype(v,mx::bfloat16),{0,0,offset,0},{1,k.shape(1),end,k.shape(3)});
  offset=end;
  return {mx::slice(*keys,{0,0,0,0},{1,k.shape(1),end,k.shape(3)}),mx::slice(*values,{0,0,0,0},{1,k.shape(1),end,k.shape(3)})};
}
void KVCache::close() { offset=0;keys.reset();values.reset(); }
ARModel::ARModel(const fs::path& directory,std::string p,bool verify):precision(std::move(p)) {
  if(precision!="bf16"&&precision!="8bit"&&precision!="4bit") throw Error("ValueError","precision must be 'bf16', '8bit', or '4bit'");
  auto manifest=verify?verify_conversion(directory):read_json(directory/"conversion.json");
  if(!manifest.is_object()||manifest.value("schema",0)!=1) throw Error("ValueError","conversion.json must be a schema-1 manifest");
  if(!manifest.contains("precisions")||!manifest["precisions"].is_object()||!manifest["precisions"].contains(precision)||!manifest["precisions"][precision].is_object()) throw Error("FileNotFoundError","Converted artifact does not contain "+precision+" AR weights");
  auto entry=manifest["precisions"][precision];std::string filename="ar-"+precision+".safetensors";
  if(entry.value("ar",std::string())!=filename) throw Error("ValueError","Manifest AR filename does not match precision");
  int bits=precision_bits(precision);
  if(!bits) { if(entry.value("dtype",std::string())!="bfloat16") throw Error("ValueError","BF16 manifest entry must declare dtype='bfloat16'"); }
  else if(entry.value("bits",0)!=bits||entry.value("group_size",0)!=64||entry.value("mode",std::string())!="affine"||entry.value("linear_only",false)!=true) throw Error("ValueError","Invalid affine quantization settings in AR manifest");
  config=read_json(directory/"config.json");
  if(!config.is_object()) throw Error("ValueError","config.json must contain a JSON object");
  const Json expected={{"hidden_size",2048},{"intermediate_size",6144},{"num_hidden_layers",28},{"num_attention_heads",16},{"num_key_value_heads",8},{"head_dim",128},{"vocab_size",VOCAB_SIZE},{"max_position_embeddings",CONTEXT},{"rope_theta",1000000.0}};
  for(auto it=expected.begin();it!=expected.end();++it) {
    auto actual=config.value(it.key(),it.key()=="max_position_embeddings"?Json(CONTEXT):Json());
    if(actual!=it.value()) throw Error("ValueError","Incompatible YuE2 config: "+it.key());
    if(it.key()!="rope_theta"&&!actual.is_number_integer()) throw Error("ValueError","Generator config field "+it.key()+" must be a positive integer");
  }
  if(config.value("tie_word_embeddings",Json(false))!=Json(false)) throw Error("ValueError","YuE2 requires untied input and output embeddings");
  if(config.contains("rope_scaling")&&!config["rope_scaling"].is_null()&&config["rope_scaling"]!=Json::object()) throw Error("ValueError","YuE2 uses fixed base RoPE without scaling");
  if(config.value("hidden_act",std::string("silu"))!="silu") throw Error("ValueError","Incompatible YuE2 config: hidden_act must be silu");
  if(!config.contains("rms_norm_eps")||!config["rms_norm_eps"].is_number()||!std::isfinite(config["rms_norm_eps"].get<double>())||config["rms_norm_eps"].get<double>()<=0) throw Error("ValueError","rms_norm_eps must be finite and positive");
  weights=std::move(mx::load_safetensors((directory/filename).string()).first);
  std::set<std::string> expected_keys;
  auto require=[&](const std::string& name,mx::Shape shape,mx::Dtype dtype) {
    expected_keys.insert(name);auto it=weights.find(name);
    if(it==weights.end()) throw Error("ValueError","Missing AR tensor: "+name);
    if(it->second.shape()!=shape||it->second.dtype()!=dtype) throw Error("ValueError","Incompatible AR tensor shape/dtype: "+name);
  };
  auto norm=[&](const std::string& n,int width) { require(n+".weight",{width},mx::bfloat16); };
  auto linear=[&](const std::string& n,int out,int in) {
    require(n+".weight",{out,bits?in*bits/32:in},bits?mx::uint32:mx::bfloat16);
    if(bits) { require(n+".scales",{out,in/64},mx::bfloat16);require(n+".biases",{out,in/64},mx::bfloat16); }
  };
  require("model.embed_tokens.weight",{VOCAB_SIZE,2048},mx::bfloat16);norm("model.norm",2048);linear("lm_head",VOCAB_SIZE,2048);
  for(int i=0;i<28;++i) {
    std::string n="model.layers."+std::to_string(i);norm(n+".input_layernorm",2048);norm(n+".post_attention_layernorm",2048);
    auto a=n+".self_attn";norm(a+".q_norm",128);norm(a+".k_norm",128);
    linear(a+".q_proj",2048,2048);linear(a+".k_proj",1024,2048);linear(a+".v_proj",1024,2048);linear(a+".o_proj",2048,2048);
    linear(n+".mlp.gate_proj",6144,2048);linear(n+".mlp.up_proj",6144,2048);linear(n+".mlp.down_proj",2048,6144);
  }
  for(const auto& [name,value]:weights) if(!expected_keys.contains(name)) throw Error("ValueError","Unexpected AR tensor: "+name);
  materialize();
}
void ARModel::materialize() const { std::vector<mx::array> tensors;tensors.reserve(weights.size());for(const auto& [name,value]:weights)tensors.push_back(value);mx::eval(tensors); }
mx::array ARModel::hidden(const mx::array& ids,std::vector<KVCache>& cache,const Cancelled& cancelled) {
  int layers=config.at("num_hidden_layers"),heads=config.at("num_attention_heads"),kv=config.at("num_key_value_heads"),dim=config.at("head_dim");
  if(ids.ndim()!=2||ids.shape(0)!=1||ids.shape(1)<1||int(cache.size())!=layers) throw Error("ValueError","AR inputs require [1, T] tokens and one cache per layer");
  int length=ids.shape(1),offset=cache.front().offset,bits=precision_bits(precision);float eps=config.at("rms_norm_eps");double theta=config.at("rope_theta");
  auto x=mx::take(weights.at("model.embed_tokens.weight"),ids,0);
  for(int i=0;i<layers;++i) {
    if(cancelled&&cancelled()) throw Error("InterruptedError","Cancelled during AR layers");
    if(cache[i].offset!=offset) throw Error("ValueError","AR cache offsets disagree");
    auto n="model.layers."+std::to_string(i),a=n+".self_attn";
    auto h=model_ops::rms_norm(x,weights.at(n+".input_layernorm.weight"),eps);
    auto q=model_ops::rms_norm(mx::reshape(model_ops::linear(h,weights,a+".q_proj",bits),{1,length,heads,dim}),weights.at(a+".q_norm.weight"),eps);
    auto k=model_ops::rms_norm(mx::reshape(model_ops::linear(h,weights,a+".k_proj",bits),{1,length,kv,dim}),weights.at(a+".k_norm.weight"),eps);
    q=model_ops::rope(mx::transpose(q,{0,2,1,3}),theta,offset);k=model_ops::rope(mx::transpose(k,{0,2,1,3}),theta,offset);
    auto v=mx::transpose(mx::reshape(model_ops::linear(h,weights,a+".v_proj",bits),{1,length,kv,dim}),{0,2,1,3});
    auto state=cache[i].update(k,v);
    float scale=float(std::pow(dim,-.5));
    auto attended=length==1?model_ops::sdpa(q,state.first,state.second,scale):
      mx::astype(mx::fast::scaled_dot_product_attention(mx::astype(q,mx::float32),mx::astype(state.first,mx::float32),mx::astype(state.second,mx::float32),scale,"causal"),q.dtype());
    x=x+model_ops::linear(mx::reshape(mx::transpose(attended,{0,2,1,3}),{1,length,heads*dim}),weights,a+".o_proj",bits);
    x=x+model_ops::mlp(model_ops::rms_norm(x,weights.at(n+".post_attention_layernorm.weight"),eps),weights,n+".mlp",bits);
  }
  return model_ops::rms_norm(x,weights.at("model.norm.weight"),eps);
}
mx::array ARModel::project(const mx::array& h,std::string_view phase) const {
  if(h.shape(-1)!=config.at("hidden_size").get<int>()) throw Error("ValueError","Hidden-state width does not match the AR model");
  int bits=precision_bits(precision);
  auto projection=[&](int start,int stop) {
    auto w=rows(weights.at("lm_head.weight"),start,stop);
    return bits?mx::quantized_matmul(h,w,rows(weights.at("lm_head.scales"),start,stop),rows(weights.at("lm_head.biases"),start,stop),true,64,bits,"affine"):mx::matmul(h,mx::transpose(w));
  };
  if(phase=="abc") return mx::astype(mx::concatenate({projection(0,EOD),projection(ABC_END,ABC_END+1)},-1),mx::bfloat16);
  if(phase=="semantic") return mx::astype(projection(MUSIC_END,CODEC_OFFSET+CODEC_SIZE),mx::bfloat16);
  throw Error("ValueError","phase must be 'abc' or 'semantic'");
}
} // namespace lyra
