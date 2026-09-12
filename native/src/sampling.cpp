#include "lyra/ar.hpp"
#include "lyra/runtime.hpp"
#include <algorithm>
#include <limits>

namespace lyra {
namespace {
int phase_size(std::string_view phase) {
  if(phase=="abc") return EOD+1;
  if(phase=="semantic") return CODEC_SIZE+1;
  throw Error("ValueError","phase must be 'abc' or 'semantic'");
}
int end_index(std::string_view phase) { phase_size(phase);return phase=="abc"?EOD:0; }
std::optional<int> native_to_local(int token,std::string_view phase) {
  if(phase=="abc") { if(token>=0&&token<EOD)return token;if(token==ABC_END)return EOD; }
  else { if(token==MUSIC_END)return 0;if(token>=CODEC_OFFSET&&token<CODEC_OFFSET+CODEC_SIZE)return token-MUSIC_END; }
  return std::nullopt;
}
int local_to_native(int token,std::string_view phase) {
  if(token<0||token>=phase_size(phase)) throw Error("ValueError","Sampled phase-local token outside vocabulary");
  return phase=="abc"?(token<EOD?token:ABC_END):MUSIC_END+token;
}
mx::array distribution(const mx::array& logits,const Sampling& sampling,const std::vector<int>& history,int step,std::string_view phase,bool legacy_off) {
  int expected=phase_size(phase);
  if(logits.shape(-1)!=expected) throw Error("ValueError","Incorrect projected phase vocabulary size");
  auto scores=mx::astype(logits,legacy_off?logits.dtype():mx::float32);
  auto infinity=mx::array(-std::numeric_limits<float>::infinity(),scores.dtype());
  if(step<sampling.min_tokens) scores=mx::where(mx::equal(mx::arange(expected),mx::array(end_index(phase))),infinity,scores);
  if(sampling.repetition_penalty!=1&&!history.empty()) {
    std::vector<int> recent;
    // Python history[-0:] selects the full history, not an empty window.
    size_t count=sampling.penalty_window==0?history.size():std::min(history.size(),size_t(sampling.penalty_window));
    recent.reserve(count);
    for(size_t i=history.size()-count;i<history.size();++i) if(auto local=native_to_local(history[i],phase)) recent.push_back(*local);
    if(!recent.empty()) {
      auto ids=mx::array(recent.data(),{int(recent.size())},mx::int32);
      auto frequency=mx::scatter_add_axis(mx::zeros({expected},scores.dtype()),ids,mx::ones(ids.shape(),scores.dtype()),0);
      auto alpha=mx::astype(mx::power(mx::array(float(sampling.repetition_penalty),mx::float32),mx::astype(frequency,mx::float32)),scores.dtype());
      scores=mx::where(mx::less(scores,mx::array(0)),mx::astype(scores*alpha,scores.dtype()),mx::astype(scores/alpha,scores.dtype()));
    }
  }
  if(sampling.temperature==0) return scores;
  if(sampling.temperature!=1) scores=mx::astype(mx::astype(scores,mx::float32)/mx::array(float(sampling.temperature),mx::float32),scores.dtype());
  int k=std::min(sampling.top_k,expected);
  auto threshold=mx::min(mx::topk(scores,k,-1),-1,true);
  scores=mx::where(mx::less(scores,threshold),infinity,scores);
  if(sampling.top_p<1) {
    auto indices=mx::argsort(-scores,-1),values=mx::take_along_axis(scores,indices,-1);
    auto probabilities=mx::softmax(values,-1,true);
    auto preceding=mx::astype(mx::cumsum(probabilities,-1)-probabilities,scores.dtype());
    auto removed=mx::logical_and(mx::greater(mx::astype(preceding,mx::float32),mx::array(float(sampling.top_p),mx::float32)),mx::greater_equal(mx::arange(expected),mx::array(legacy_off?3:1)));
    values=mx::where(removed,infinity,values);
    scores=mx::put_along_axis(values,indices,values,-1);
  }
  return scores;
}
mx::array guided_logits(const mx::array& conditional,const std::optional<mx::array>& unconditional,double scale) {
  if(!unconditional) return conditional;
  if(conditional.shape()!=unconditional->shape()||conditional.dtype()!=unconditional->dtype()) throw Error("ValueError","Conditional and unconditional logits must have the same shape and dtype");
  auto dtype=conditional.dtype();auto delta=mx::astype(conditional-*unconditional,dtype);
  auto scaled=mx::astype(mx::astype(delta,mx::float32)*mx::array(float(scale),mx::float32),dtype);
  return mx::astype(*unconditional+scaled,dtype);
}
class RequestRNG {
  mx::array key;
 public:
  explicit RequestRNG(uint64_t seed):key(std::initializer_list<uint32_t>{uint32_t(seed>>32),uint32_t(seed)},mx::uint32) {
    if(seed>=(uint64_t(1)<<63)) throw Error("ValueError","seed must be an integer in [0, 2**63)");
  }
  mx::array categorical(const mx::array& logits) {
    auto keys=mx::random::split(key);key=keys.first;
    auto probabilities=mx::softmax(logits,-1,true);
    return mx::random::categorical(mx::log(mx::astype(probabilities,mx::float32)),-1,keys.second);
  }
};
void cancelled_at(const Cancelled& cancelled,const std::string& message) {
  if(cancelled&&cancelled()) throw Error("InterruptedError",message);
}
void validate_tokens(const std::vector<int>& ids,int vocab) {
  if(ids.empty()) throw Error("ValueError","AR prefix must contain at least one token");
  for(int id:ids) if(id<0||id>=vocab) throw Error("ValueError","Token IDs must be in [0, "+std::to_string(vocab)+")");
}
mx::array last_hidden(const mx::array& h) {
  return mx::reshape(mx::slice(h,{0,h.shape(1)-1,0},{1,h.shape(1),h.shape(2)}),{1,h.shape(2)});
}
mx::array prefill(ARModel& model,const std::vector<int>& tokens,int capacity,std::string_view phase,std::vector<KVCache>& cache,const Cancelled& cancelled) {
  cancelled_at(cancelled,"Cancelled before prefill");
  int layers=model.config.at("num_hidden_layers");cache.reserve(layers);for(int i=0;i<layers;++i)cache.emplace_back(capacity);
  auto inputs=mx::array(tokens.data(),{1,int(tokens.size())},mx::int32);
  std::optional<mx::array> hidden;
  for(int start=0;start<int(tokens.size());start+=1024) {
    cancelled_at(cancelled,"Cancelled during prefill");
    int stop=std::min(start+1024,int(tokens.size()));
    hidden=model.hidden(mx::slice(inputs,{0,start},{1,stop}),cache,cancelled);
    if(stop<int(tokens.size())) {
      std::vector<mx::array> state;state.reserve(2*cache.size());
      for(const auto& c:cache) {state.push_back(*c.keys);state.push_back(*c.values);}
      mx::eval(state);hidden.reset();mx::clear_cache();
    }
  }
  return model.project(last_hidden(*hidden),phase);
}
}
TokenGeneration generate_tokens(ARModel& model,const std::vector<int>& prefix,const Sampling& sampling,uint64_t seed,std::string_view phase,const std::vector<int>& negative,double cfg_scale,bool legacy_off,const Cancelled& cancelled,const TokenCallback& on_token) {
  phase_size(phase);sampling.validate();validate_tokens(prefix,model.config.at("vocab_size"));
  if(prefix.size()+size_t(sampling.max_tokens)>CONTEXT) throw Error("ValueError","Prefix + requested generation budget exceeds 24576; no implicit truncation");
  if(cfg_scale!=1&&negative.empty()) throw Error("ValueError","CFG requires a negative prefix");
  if(!negative.empty()) {
    validate_tokens(negative,model.config.at("vocab_size"));
    if(negative.size()+size_t(sampling.max_tokens)>CONTEXT) throw Error("ValueError","Negative prefix + generation budget exceeds context");
  }
  cancelled_at(cancelled,"Cancelled before prefill");RequestRNG rng(seed);
  // Cache owners are request-local; optional backing arrays release on every exit.
  std::vector<KVCache> positive_cache,negative_cache;
  mx::synchronize();double started=monotonic_seconds();
  auto conditional=prefill(model,prefix,int(prefix.size())+sampling.max_tokens,phase,positive_cache,cancelled);
  std::optional<mx::array> unconditional;
  if(cfg_scale!=1) unconditional=prefill(model,negative,int(negative.size())+sampling.max_tokens,phase,negative_cache,cancelled);
  if(unconditional)mx::eval({conditional,*unconditional});else mx::eval(conditional);
  mx::synchronize();double prefill_seconds=monotonic_seconds()-started;
  TokenGeneration result;std::optional<double> first;bool eos=false;int end=end_index(phase);
  result.tokens.reserve(sampling.max_tokens);
  for(int step=0;step<sampling.max_tokens;++step) {
    cancelled_at(cancelled,"Cancelled during "+std::string(phase));
    auto logits=guided_logits(conditional,unconditional,cfg_scale);
    auto scores=distribution(logits,sampling,result.tokens,step,phase,legacy_off);
    auto next=sampling.temperature==0?mx::argmax(scores,-1):rng.categorical(scores);
    int local=next.dtype()==mx::uint32?int(next.item<uint32_t>()):next.item<int32_t>();
    int token=local_to_native(local,phase);
    if(!first)first=monotonic_seconds()-started;
    if(on_token)on_token(phase,token);
    if(local==end) {eos=true;break;}
    result.tokens.push_back(token);
    if(step+1<sampling.max_tokens) {
      auto input=mx::array(&token,{1,1},mx::int32);
      conditional=model.project(last_hidden(model.hidden(input,positive_cache,cancelled)),phase);
      if(!negative_cache.empty())unconditional=model.project(last_hidden(model.hidden(input,negative_cache,cancelled)),phase);
    }
  }
  mx::synchronize();double seconds=monotonic_seconds()-started;size_t count=result.tokens.size()+size_t(eos);
  result.timing={{"seconds",seconds},{"prefill_seconds",prefill_seconds},{"ttft_seconds",first?Json(*first):Json()},{"output_tokens",count},{"content_tokens",result.tokens.size()},{"output_tps",count/seconds},{"prefix_tokens",prefix.size()},{"cfg_branches",cfg_scale==1?1:2},{"execution","eager"},{"attention","sdpa"},{"backend","mlx"}};
  result.truncated=!eos;return result;
}
} // namespace lyra
