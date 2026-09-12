#include "lyra/protocol.hpp"
#include "lyra/storage.hpp"
#import <Foundation/Foundation.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <queue>
#include <sstream>
#include <unordered_map>

namespace lyra {
namespace {
[[noreturn]] void invalid(const std::string& s) { throw Error("ValueError",s); }
void object(const Json& j) { if (!j.is_object()) throw Error("TypeError","Expected a dictionary"); }
void keys(const Json& j,std::initializer_list<std::string_view> allowed,const char* type) {
  object(j);
  for(auto it=j.begin();it!=j.end();++it)
    if(std::find(allowed.begin(),allowed.end(),it.key())==allowed.end())
      throw Error("TypeError",std::string(type)+" got an unexpected keyword argument '"+it.key()+"'");
}
bool integer(const Json& j) { return j.is_number_integer() && !j.is_boolean(); }
int count(const Json& j,const char* error) {
  if(!integer(j) || (j.is_number_unsigned() && j.get<uint64_t>()>uint64_t(INT_MAX)) ||
     (!j.is_number_unsigned() && (j.get<int64_t>()<INT_MIN || j.get<int64_t>()>INT_MAX))) invalid(error);
  return j.get<int>();
}
double number(const Json& j) {
  if(j.is_boolean()) return j.get<bool>() ? 1. : 0.;
  if(!j.is_number()) throw Error("TypeError","must be real number");
  return j.get<double>();
}
Json numeric_representation(const Json& values,const char* key,double current) {
  if(values.contains(key)) {
    const auto& value=values.at(key);
    if((value.is_number()||value.is_boolean()) && number(value)==current) return value;
  }
  return current;
}
std::string string(const Json& j,const char* message) {
  if(!j.is_string()) throw Error("TypeError",message);
  return j.get<std::string>();
}
NSString* utf8(std::string_view s) {
  NSString* value=[[NSString alloc] initWithBytes:s.data() length:s.size() encoding:NSUTF8StringEncoding];
  if(!value) throw Error("UnicodeDecodeError","Invalid UTF-8 text");
  return value;
}
std::string bytes(NSString* s) {
  NSData* d=[s dataUsingEncoding:NSUTF8StringEncoding];
  return std::string(static_cast<const char*>(d.bytes),d.length);
}
bool nonwhite(std::string_view s) {
  @autoreleasepool {
    NSString* value=utf8(s);
    NSMutableCharacterSet* ws=[[NSCharacterSet whitespaceAndNewlineCharacterSet] mutableCopy];
    [ws addCharactersInRange:NSMakeRange(0x1c,4)];
    return [value rangeOfCharacterFromSet:[ws invertedSet]].location!=NSNotFound;
  }
}
void abc_valid(const std::vector<int>& ids) {
  for(int id:ids) if(id<0 || id>=EOD) invalid("ABC IDs must remain inside the ordinary text vocabulary");
}
// Unicode's maximal-subpart replacement, matching Python bytes.decode(errors="replace").
std::string replacement_utf8(std::string_view s) {
  std::string out; out.reserve(s.size());
  for(size_t i=0;i<s.size();) {
    const auto c=static_cast<unsigned char>(s[i]);
    if(c<0x80) {out+=s[i++];continue;}
    int n=c>=0xc2&&c<=0xdf?2:c>=0xe0&&c<=0xef?3:c>=0xf0&&c<=0xf4?4:0;
    size_t end=i+1;
    if(n) {
      for(int k=1;k<n && end<s.size();++k) {
        auto d=static_cast<unsigned char>(s[end]);
        if(d<0x80 || d>0xbf || (k==1 && ((c==0xe0&&d<0xa0)||(c==0xed&&d>0x9f)||(c==0xf0&&d<0x90)||(c==0xf4&&d>0x8f)))) break;
        ++end;
      }
    }
    if(n && end-i==static_cast<size_t>(n)) out.append(s.substr(i,n)); else out+="\xef\xbf\xbd";
    i=end;
  }
  return out;
}
}

void Sampling::validate() const {
  if(!std::isfinite(temperature)||!std::isfinite(top_p)||!std::isfinite(repetition_penalty)) invalid("Sampling numbers must be finite");
  if(!(temperature>=0&&temperature<=5&&top_p>0&&top_p<=1&&top_k>=1)) invalid("Invalid sampling temperature/top_p/top_k");
  if(repetition_penalty<=0 || penalty_window<1 || penalty_window>100) invalid("Invalid repetition penalty/window");
  if(min_tokens<0 || min_tokens>max_tokens || max_tokens<1) invalid("Require 0 <= min_tokens <= max_tokens");
}
Json Sampling::to_json() const { return {{"temperature",numeric_representation(numeric_values,"temperature",temperature)},{"top_p",numeric_representation(numeric_values,"top_p",top_p)},{"top_k",top_k},{"repetition_penalty",numeric_representation(numeric_values,"repetition_penalty",repetition_penalty)},{"penalty_window",penalty_window},{"min_tokens",min_tokens},{"max_tokens",max_tokens}}; }
Sampling Sampling::from_json(const Json& j,const Sampling& defaults) {
  if(j.is_null()) return defaults;
  keys(j,{"temperature","top_p","top_k","repetition_penalty","penalty_window","min_tokens","max_tokens"},"Sampling");
  Sampling s=defaults;
  if(j.contains("temperature")) s.temperature=number(j.at("temperature"));
  if(j.contains("top_p")) s.top_p=number(j.at("top_p"));
  if(j.contains("repetition_penalty")) s.repetition_penalty=number(j.at("repetition_penalty"));
  if(j.contains("top_k")) s.top_k=count(j.at("top_k"),"Sampling counts must be integers");
  if(j.contains("penalty_window")) s.penalty_window=count(j.at("penalty_window"),"Sampling counts must be integers");
  if(j.contains("min_tokens")) s.min_tokens=count(j.at("min_tokens"),"Sampling counts must be integers");
  if(j.contains("max_tokens")) s.max_tokens=count(j.at("max_tokens"),"Sampling counts must be integers");
  for(const char* key:{"temperature","top_p","repetition_penalty"}) if(j.contains(key)) s.numeric_values[key]=j.at(key);
  s.validate(); return s;
}
void GenerationConfig::validate() const {
  abc.validate(); semantic.validate();
  if(context!=CONTEXT || ode_method!="midpoint" || ode_steps<1) invalid("Require context=24576 and midpoint with positive integer steps");
}
Json GenerationConfig::to_json() const {
  Json serialized_context=context;
  if(context_value.is_number() && number(context_value)==context) serialized_context=context_value;
  return {{"abc",abc.to_json()},{"semantic",semantic.to_json()},{"ode_steps",ode_steps},{"ode_method",ode_method},{"context",serialized_context},{"version",version}};
}
GenerationConfig GenerationConfig::from_json(const Json& j) {
  keys(j,{"abc","semantic","ode_steps","ode_method","context","version"},"GenerationConfig");
  GenerationConfig c;
  if(j.contains("abc")) c.abc=Sampling::from_json(j.at("abc"),c.abc);
  if(j.contains("semantic")) c.semantic=Sampling::from_json(j.at("semantic"),c.semantic);
  constexpr auto error="Require context=24576 and midpoint with positive integer steps";
  if(j.contains("ode_steps")) c.ode_steps=count(j.at("ode_steps"),error);
  if(j.contains("context")) {if(!j.at("context").is_number() || number(j.at("context"))!=CONTEXT) invalid(error);c.context=CONTEXT;c.context_value=j.at("context");}
  if(j.contains("ode_method")) {if(j.at("ode_method")!="midpoint") invalid(error);c.ode_method="midpoint";}
  if(j.contains("version")) c.version=j.at("version");
  c.validate(); return c;
}
std::string_view instruction(std::string_view cot) {
  if(cot=="off") return "Generate music with codec tokens from the given conditions.";
  if(cot=="melody") return "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.";
  if(cot=="full") return "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.";
  invalid("cot must be off, melody or full");
}
void SongRequest::validate() const {
  instruction(cot);
  if(seed>=uint64_t(1)<<63) invalid("seed must be an integer in [0, 2**63)");
  auto alnum=[](unsigned char c){return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9');};
  if(id.empty()||id.size()>180||!alnum(id[0])||!std::all_of(id.begin(),id.end(),[&](unsigned char c){return alnum(c)||c=='_'||c=='.'||c=='-';})) invalid("id must be a filename-safe identifier");
  if(abc && (cot=="off"||!nonwhite(*abc))) invalid("External ABC requires nonempty text and cot=melody/full");
  if(cfg_scale && (!std::isfinite(*cfg_scale)||*cfg_scale<0||*cfg_scale>20)) invalid("cfg_scale must be finite and in [0,20]");
}
double SongRequest::guidance() const {return cfg_scale.value_or(cot=="off"?1.01:1.);}
std::string SongRequest::text() const {return std::string(instruction(cot))+"\n[Tags]\n"+style+"\n[Lyrics]\n"+lyrics+"\n";}
Json SongRequest::to_json() const {return {{"style",style},{"lyrics",lyrics},{"cot",cot},{"seed",seed},{"abc",abc?Json(*abc):Json(nullptr)},{"cfg_scale",cfg_scale?numeric_representation(numeric_values,"cfg_scale",*cfg_scale):Json(nullptr)},{"id",id}};}
SongRequest SongRequest::from_json(const Json& j) {
  keys(j,{"style","tags","lyrics","cot","seed","abc","cfg_scale","id","abc_sampling","semantic_sampling"},"SongRequest");
  SongRequest r;
  Json style=j.value("style",Json(nullptr)),tags=j.value("tags",Json(nullptr));
  if(!style.is_null()&&!tags.is_null()&&style!=tags) invalid("style and tags are aliases and cannot disagree");
  if(style.is_null()) style=tags;
  if(style.is_null()||!j.contains("lyrics")||j.at("lyrics").is_null()) invalid("Provide style and lyrics");
  if(j.contains("cot")) {if(!j.at("cot").is_string()) invalid("cot must be off, melody or full");r.cot=j.at("cot").get<std::string>();}
  instruction(r.cot);
  r.style=string(style,"style and lyrics must be strings");r.lyrics=string(j.at("lyrics"),"style and lyrics must be strings");
  if(j.contains("seed")) {
    const auto& s=j.at("seed");
    if(!integer(s)||(!s.is_number_unsigned()&&s.get<int64_t>()<0)||s.get<uint64_t>()>=(uint64_t(1)<<63)) invalid("seed must be an integer in [0, 2**63)");
    r.seed=s.get<uint64_t>();
  }
  if(j.contains("id")) r.id=string(j.at("id"),"expected string or bytes-like object");
  if(j.contains("abc")&&!j.at("abc").is_null()) {if(!j.at("abc").is_string()) invalid("External ABC requires nonempty text and cot=melody/full");r.abc=j.at("abc").get<std::string>();}
  if(j.contains("cfg_scale")&&!j.at("cfg_scale").is_null()) {r.cfg_scale=number(j.at("cfg_scale"));r.numeric_values["cfg_scale"]=j.at("cfg_scale");}
  r.validate();return r;
}
void FloatMatrix::validate(std::string_view name,int64_t columns) const {
  if(rows<1||cols<1||(columns && cols!=columns)||uint64_t(rows)>std::numeric_limits<size_t>::max()/uint64_t(cols)||values.size()!=uint64_t(rows)*uint64_t(cols)) invalid(std::string(name)+" must be a nonempty matrix with the expected shape");
  if(!std::all_of(values.begin(),values.end(),[](float x){return std::isfinite(x);})) invalid(std::string(name)+" must contain finite values");
}
std::vector<int> token_prefixes(const SongRequest& r,const Tokenizer& t,const std::optional<std::vector<int>>& supplied) {
  std::vector<int> out{EOD};auto text=t.encode(r.text());out.insert(out.end(),text.begin(),text.end());out.push_back(ABC_START);
  if(r.cot=="off") {out.insert(out.end(),{ABC_END,MUSIC_START});return out;}
  if(!supplied&&!r.abc) return out;
  const auto generated=!supplied?t.encode(*r.abc):std::vector<int>{};const auto& ids=supplied?*supplied:generated;
  abc_valid(ids);out.insert(out.end(),ids.begin(),ids.end());out.insert(out.end(),{ABC_END,MUSIC_START});return out;
}
std::vector<int> negative_prefix(const SongRequest& r,const Tokenizer& t,const std::optional<std::vector<int>>& ids) {
  std::vector<int> out{EOD};auto text=t.encode(instruction(r.cot));out.insert(out.end(),text.begin(),text.end());
  if(r.cot=="off") {out.push_back(MUSIC_START);return out;}
  if(!ids) invalid("Symbolic CFG must retain the exact positive-branch ABC IDs");
  abc_valid(*ids);out.push_back(ABC_START);out.insert(out.end(),ids->begin(),ids->end());out.insert(out.end(),{ABC_END,MUSIC_START});return out;
}
std::vector<std::pair<int,int>> chunk_ranges(int frames,int prefix,int context) {
  int64_t available=int64_t(context)-prefix-3;
  if(frames<1||available<2) invalid("Empty codec or prefix leaves no acoustic context");
  int64_t size=std::min<int64_t>(available/2,CONTEXT);
  std::vector<std::pair<int,int>> out;out.reserve((int64_t(frames)+size-1)/size);
  for(int64_t a=0;a<frames;a+=size) out.emplace_back(a,std::min(a+size,int64_t(frames)));
  return out;
}
Json request_kwargs(Json data,const fs::path& base) {
  object(data);
  static const std::vector<std::string> allowed={"style","tags","lyrics","cot","seed","abc","cfg_scale","id","abc_sampling","semantic_sampling"};
  static const std::vector<std::string> meta={"lang","eval_index","clip_id","prompt","abc_path"};
  Json result=Json::object();std::vector<std::string> unknown;
  for(auto it=data.begin();it!=data.end();++it) {
    if(std::find(allowed.begin(),allowed.end(),it.key())!=allowed.end()) result[it.key()]=it.value();
    else if(std::find(meta.begin(),meta.end(),it.key())==meta.end()) unknown.push_back(it.key());
  }
  if(!unknown.empty()) {std::string msg="Unknown request fields: [";for(size_t i=0;i<unknown.size();++i){if(i)msg+=", ";msg+="'"+unknown[i]+"'";}invalid(msg+"]");}
  if(data.contains("abc_path")) {
    if(data.contains("abc")&&!data.at("abc").is_null()) invalid("Pass abc or abc_path, not both");
    auto text=read_text(base/fs::path(string(data.at("abc_path"),"abc_path must be a path string")));
    @autoreleasepool { (void)utf8(text); }
    result["abc"]=std::move(text);
  }
  if(data.contains("prompt")&&!data.at("prompt").is_null()) {
    if(!result.contains("lyrics")) throw Error("KeyError","'lyrics'");
    SongRequest request;
    auto cot=result.value("cot",Json("full"));
    if(!cot.is_string()) invalid("cot must be off, melody or full");
    request.cot=cot.get<std::string>();
    instruction(request.cot);
    request.style=string(result.contains("style")?result.at("style"):result.value("tags",Json(nullptr)),"style and lyrics must be strings");
    request.lyrics=string(result.at("lyrics"),"style and lyrics must be strings");
    request.validate();
    if(data.at("prompt")!=request.text()) invalid("Historical literal prompt does not match native instruction/style/lyrics");
  }
  return result;
}

struct Tokenizer::Impl {
  struct ByteHash {
    using is_transparent=void;
    size_t operator()(std::string_view value) const noexcept {return std::hash<std::string_view>{}(value);}
  };
  std::unordered_map<std::string,int,ByteHash,std::equal_to<>> ranks;
  std::vector<std::string> pieces;
  NSRegularExpression* regex;
  explicit Impl(const fs::path& path):pieces(EOD+208) {
    @autoreleasepool {
      std::istringstream input(read_text(path));std::string line;
      while(std::getline(input,line)) {
        if(line.empty()) continue;
        std::istringstream row(line);std::string encoded,extra;int rank;
        if(!(row>>encoded>>rank)||(row>>extra)||rank<0||rank>=EOD) invalid("Invalid checkpoint-native qwen.tiktoken vocabulary");
        NSData* decoded=[[NSData alloc] initWithBase64EncodedString:utf8(encoded) options:0];
        if(!decoded||!decoded.length) invalid("Invalid base64 vocabulary token");
        std::string piece(static_cast<const char*>(decoded.bytes),decoded.length);
        ranks[piece]=rank;pieces[rank]=std::move(piece);
      }
      if(ranks.size()!=EOD||std::any_of(pieces.begin(),pieces.begin()+EOD,[](const auto& s){return s.empty();})) invalid("Expected checkpoint-native qwen.tiktoken (151643 ordinary tokens)");
      const char* specials[]={"<|endoftext|>","<|im_start|>","<|im_end|>","<R>","<S>","<X>","<mask>","<sep>"};
      for(int i=0;i<8;++i)pieces[EOD+i]=specials[i];
      for(int i=0;i<200;++i)pieces[EOD+8+i]="<extra_"+std::to_string(i)+">";
      pieces[ABC_START]="<abc>";pieces[ABC_END]="</abc>";
      NSError* error=nil;
      regex=[[NSRegularExpression alloc] initWithPattern:@"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" options:0 error:&error];
      if(!regex) throw Error("RuntimeError",bytes(error.localizedDescription));
    }
  }
  void bpe(std::string_view word,std::vector<int>& out) const {
    auto direct=ranks.find(word);if(direct!=ranks.end()){out.push_back(direct->second);return;}
    struct Node {size_t start,end;int prev,next;uint64_t version=0;bool live=true;};
    struct Pair {int rank,left,right;uint64_t lv,rv;};
    struct Later {bool operator()(const Pair& a,const Pair& b) const{return a.rank!=b.rank?a.rank>b.rank:a.left>b.left;}};
    std::vector<Node> nodes;nodes.reserve(word.size());
    for(size_t i=0;i<word.size();++i)nodes.push_back({i,i+1,int(i)-1,i+1<word.size()?int(i+1):-1});
    std::priority_queue<Pair,std::vector<Pair>,Later> queue;
    auto add=[&](int l){if(l<0)return;auto& a=nodes[l];int r=a.next;if(r<0)return;auto& b=nodes[r];auto found=ranks.find(word.substr(a.start,b.end-a.start));if(found!=ranks.end())queue.push({found->second,l,r,a.version,b.version});};
    for(size_t i=0;i+1<nodes.size();++i)add(int(i));
    while(!queue.empty()) {
      auto p=queue.top();queue.pop();auto& a=nodes[p.left];auto& b=nodes[p.right];
      if(!a.live||!b.live||a.next!=p.right||a.version!=p.lv||b.version!=p.rv)continue;
      a.end=b.end;a.next=b.next;++a.version;b.live=false;
      if(b.next>=0)nodes[b.next].prev=p.left;
      add(a.prev);add(p.left);
    }
    for(int i=nodes.empty()?-1:0;i>=0;i=nodes[i].next) {
      auto& n=nodes[i];auto found=ranks.find(word.substr(n.start,n.end-n.start));
      if(found==ranks.end()) invalid("Vocabulary is missing a byte token");out.push_back(found->second);
    }
  }
};
Tokenizer::Tokenizer(const fs::path& p):impl_(std::make_unique<Impl>(p)){}
Tokenizer::~Tokenizer()=default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept=default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept=default;
std::vector<int> Tokenizer::encode(std::string_view text) const {
  @autoreleasepool {
    NSString* normalized=[utf8(text) precomposedStringWithCanonicalMapping];
    std::vector<int> out;
    // Enumerating incrementally bounds autoreleased substring/UTF-8 data lifetime.
    NSUInteger offset=0;
    while(offset<normalized.length) {
      @autoreleasepool {
        NSTextCheckingResult* match=[impl_->regex firstMatchInString:normalized options:0 range:NSMakeRange(offset,normalized.length-offset)];
        if(!match||match.range.location!=offset||!match.range.length) throw Error("RuntimeError","Tokenizer regex did not cover input");
        impl_->bpe(bytes([normalized substringWithRange:match.range]),out);
        offset=NSMaxRange(match.range);
      }
    }
    return out;
  }
}
std::string Tokenizer::decode(const std::vector<int>& tokens) const {
  std::string raw;size_t size=0;for(int t:tokens)if(t>=0&&t<int(impl_->pieces.size()))size+=impl_->pieces[t].size();raw.reserve(size);
  for(int t:tokens)if(t>=0&&t<int(impl_->pieces.size()))raw+=impl_->pieces[t];
  return replacement_utf8(raw);
}
} // namespace lyra
