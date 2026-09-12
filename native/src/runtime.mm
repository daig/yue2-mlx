#include "lyra/runtime.hpp"
#include "lyra/storage.hpp"
#include <mlx/mlx.h>
#include <mlx/backend/metal/metal.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <IOKit/ps/IOPowerSources.h>
#include <IOKit/ps/IOPSKeys.h>
#include <mach/mach.h>
#include <mach-o/dyld.h>
#include <libproc.h>
#include <sys/sysctl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/utsname.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>
#include <algorithm>

namespace lyra {
namespace {
constexpr uint64_t GiB=1ULL<<30, MiB=1ULL<<20;
volatile sig_atomic_t interrupted=0;
std::atomic<bool> initialized=false;
std::once_flag init_once;
std::recursive_mutex gpu_mutex;
std::vector<std::pair<const void*,double>> gpu_owners;
std::vector<ResourceMonitor*> gpu_monitors;
int gpu_fd=-1;
std::mutex stderr_mutex;
void interrupt_handler(int signal) { interrupted=signal; }
[[noreturn]] void os_error(const std::string& operation) { throw Error("OSError",operation+": "+std::strerror(errno)); }
template<class T> T sysctl_value(const char* name) {
  T value{}; size_t bytes=sizeof(value);
  if(sysctlbyname(name,&value,&bytes,nullptr,0)!=0) os_error(name);
  if(bytes!=sizeof(value)) throw Error("OSError",std::string(name)+": unexpected result size");
  return value;
}
std::string error_text(std::exception_ptr error) {
  if(!error) return {};
  try { std::rethrow_exception(error); }
  catch(const std::exception& e) { return exception_type(e)+": "+e.what(); }
  catch(...) { return "RuntimeError: Unknown native exception"; }
}
void diagnostic(const std::string& text) noexcept {
  std::lock_guard lock(stderr_mutex); std::fprintf(stderr,"lyra: %s\n",text.c_str());
}
int exclusive_file(const fs::path& path) {
  if(!path.parent_path().empty()) fs::create_directories(path.parent_path());
  int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);
  if(fd<0) { if(errno==EEXIST) throw Error("FileExistsError","Resource evidence already exists: "+path.string()); os_error("Creating resource evidence"); }
  return fd;
}
void write_fd(int fd,std::string_view data) {
  while(!data.empty()) { auto n=::write(fd,data.data(),data.size()); if(n<0) { if(errno==EINTR) continue; os_error("Writing resource evidence"); } if(!n) throw Error("OSError","Zero-byte resource evidence write"); data.remove_prefix(n); }
}
std::string ascii(std::string value) { for(auto& c:value) if(c<' '||c>'~') c=' '; return value; }
const fs::path& executable_path() {
  static const fs::path path=[] {
    uint32_t size=0; _NSGetExecutablePath(nullptr,&size); std::vector<char> bytes(size);
    if(_NSGetExecutablePath(bytes.data(),&size)) throw Error("OSError","Cannot locate native executable");
    return fs::canonical(bytes.data());
  }();
  return path;
}
const fs::path& metallib_path() {
  static const fs::path path=executable_path().parent_path().parent_path()/"lib"/"mlx.metallib";
  return path;
}
void configure_metallib() {
  static std::once_flag configured;
  std::call_once(configured,[] {
    if(!fs::is_regular_file(metallib_path())) throw Error("FileNotFoundError","Required native MLX kernel asset is missing: "+metallib_path().string());
    mlx::core::metal::set_metallib_path(metallib_path().string());
  });
}
const std::string& executable_hash() {static const std::string hash=sha256_file(executable_path());return hash;}
const std::string& metallib_hash() {static const std::string hash=sha256_file(metallib_path());return hash;}
}

double monotonic_seconds() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
bool cancellation_requested() { return interrupted!=0; }
void check_cancelled() { if(cancellation_requested()) throw Error("InterruptedError","Execution interrupted by signal "+std::to_string(interrupted)); }
void check_metal_allocation(uint64_t bytes) {
  check_cancelled();
  std::unique_lock lock(gpu_mutex, std::try_to_lock);
  if (!lock.owns_lock() || gpu_owners.empty())
    throw Error("RuntimeError", "Metal allocation requires the active GPU execution guard");
  for (auto* monitor : gpu_monitors) monitor->check();
  double budget = gpu_owners.front().second;
  for (const auto& owner : gpu_owners) budget = std::min(budget, owner.second);
  const uint64_t limit = static_cast<uint64_t>((budget - 1) * GiB);
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) throw Error("RuntimeError", "Metal device unavailable during allocation");
    const uint64_t allocated = device.currentAllocatedSize;
    if (allocated > limit || bytes > limit - allocated)
      throw Error("MemoryError", "Metal allocation exceeds the process GPU budget");
  }
}
void initialize_runtime() {
  const char* precision=std::getenv("MLX_ENABLE_TF32");
  if(precision && std::string_view(precision)!="0") throw Error("RuntimeError","Set MLX_ENABLE_TF32=0 before using MLX for faithful attention");
  std::call_once(init_once,[]{
    const char* flag=std::getenv("MLX_ENABLE_TF32");
    if(flag && std::string_view(flag)!="0") throw Error("RuntimeError","Set MLX_ENABLE_TF32=0 before using MLX for faithful attention");
    if(!flag && setenv("MLX_ENABLE_TF32","0",1)) os_error("Setting MLX_ENABLE_TF32");
    struct sigaction action{}; action.sa_handler=interrupt_handler; sigemptyset(&action.sa_mask);
    if(sigaction(SIGINT,&action,nullptr)||sigaction(SIGTERM,&action,nullptr)) os_error("Installing cancellation handlers");
    @autoreleasepool {
      auto version=[[NSProcessInfo processInfo] operatingSystemVersion];
      if(version.majorVersion<26||(version.majorVersion==26&&version.minorVersion<2)) throw Error("RuntimeError","The native GPU runtime requires macOS 26.2 or newer");
    }
    configure_metallib();
    if(!mlx::core::metal::is_available()) throw Error("RuntimeError","MLX Metal is required; CPU fallback is not supported");
    mlx::core::set_default_device(mlx::core::Device::gpu);
    initialized.store(true,std::memory_order_release);
  });
}
Json runtime_info() {
  const char* flag=std::getenv("MLX_ENABLE_TF32");
  if(!flag && setenv("MLX_ENABLE_TF32","0",1)) os_error("Setting MLX_ENABLE_TF32");
  struct utsname system{}; if(uname(&system)) os_error("uname");
  @autoreleasepool {
    auto version=[[NSProcessInfo processInfo] operatingSystemVersion];
    Json device=Json::object(),errors=Json::object(),unsafe=Json::object(),kernel_hash=nullptr;
    bool available=false;
    try {configure_metallib();kernel_hash=metallib_hash();}
    catch(const std::exception& e) {errors["mlx_metallib"]=e.what();}
    try {
      available=errors.empty() && mlx::core::metal::is_available();
      if(available) for(const auto& [key,value]:mlx::core::device_info(mlx::core::Device(mlx::core::Device::gpu))) std::visit([&](const auto& v){device[key]=v;},value);
      else errors["mlx_metal"]="MLX Metal is required; CPU fallback is not supported";
    } catch(const std::exception& e) {errors["mlx_metal"]=e.what();}
    if(std::string_view(std::getenv("MLX_ENABLE_TF32"))!="0") unsafe["MLX_ENABLE_TF32"]=std::getenv("MLX_ENABLE_TF32");
    return {{"lyra",LYRA_VERSION},{"mlx",mlx::core::version()},{"implementation","native C++20/Objective-C++"},{"python_dependency",false},
      {"system",system.sysname},{"platform",system.sysname},{"machine",system.machine},{"architecture",system.machine},
      {"macos",std::to_string(version.majorVersion)+"."+std::to_string(version.minorVersion)+"."+std::to_string(version.patchVersion)},
      {"supported_os",version.majorVersion>26||(version.majorVersion==26&&version.minorVersion>=2)},{"supported_arch",std::string_view(system.machine)=="arm64"},
      {"unsafe_environment",unsafe},{"versions",{{"lyra",LYRA_VERSION},{"mlx",mlx::core::version()}}},{"backends",{{"mlx_metal",available}}},
      {"errors",errors},{"dependencies_ready",errors.empty()},{"backend","mlx-metal"},{"device",device},{"MLX_ENABLE_TF32",std::getenv("MLX_ENABLE_TF32")},
      {"executable_sha256",executable_hash()},{"metallib_sha256",kernel_hash},
      {"runtime_sha256",kernel_hash.is_null()?Json(nullptr):Json(identity({{"executable_sha256",executable_hash()},{"metallib_sha256",kernel_hash}}))}};
  }
}
Json power_source() {
  CFTypeRef info=IOPSCopyPowerSourcesInfo();
  if(!info) throw Error("OSError","IOPSCopyPowerSourcesInfo failed");
  CFStringRef source=IOPSGetProvidingPowerSourceType(info);
  if(!source) { CFRelease(info); throw Error("OSError","IOPSGetProvidingPowerSourceType failed"); }
  bool ac=CFEqual(source,CFSTR(kIOPSACPowerValue));
  char text[256]; bool converted=CFStringGetCString(source,text,sizeof(text),kCFStringEncodingUTF8);
  CFRelease(info);
  if(!converted) throw Error("OSError","Cannot decode power source");
  return {{"ac_connected",ac},{"source",text},{"source_api","IOPSGetProvidingPowerSourceType"},{"pmset",nullptr}};
}
Json memory_snapshot() {
  vm_statistics64_data_t vm{}; mach_msg_type_number_t count=HOST_VM_INFO64_COUNT;
  mach_port_t host=mach_host_self(); auto status=host_statistics64(host,HOST_VM_INFO64,reinterpret_cast<host_info64_t>(&vm),&count); mach_port_deallocate(mach_task_self(),host);
  // New SDKs append fields that older supported kernels do not return. All
  // counters used here are present through the revision-1 structure.
  if(status!=KERN_SUCCESS || count<HOST_VM_INFO64_REV1_COUNT) throw Error("OSError","host_statistics64 failed: status="+std::to_string(status)+", count="+std::to_string(count));
  rusage_info_v4 usage{};
  if(proc_pid_rusage(getpid(),RUSAGE_INFO_V4,reinterpret_cast<rusage_info_t*>(&usage))) os_error("proc_pid_rusage");
  auto swap=sysctl_value<xsw_usage>("vm.swapusage"); auto total=sysctl_value<uint64_t>("hw.memsize");
  auto pressure=sysctl_value<int>("kern.memorystatus_vm_pressure_level");
  uint64_t page=static_cast<uint64_t>(getpagesize());
  // psutil Darwin available includes inactive and all free (including speculative) pages.
  Json result={{"rss_bytes",usage.ri_resident_size},{"physical_footprint_bytes",usage.ri_phys_footprint},{"process_lifetime_peak_footprint_bytes",usage.ri_lifetime_max_phys_footprint},
    {"system_total_bytes",total},{"system_available_bytes",(uint64_t(vm.free_count)+vm.inactive_count)*page},{"system_swap_used_bytes",swap.xsu_used},
    {"system_swap_in_bytes",vm.swapins*page},{"system_swap_out_bytes",vm.swapouts*page},{"system_memory_pressure_level",pressure},
    {"system_free_bytes",uint64_t(vm.free_count)*page},{"system_wired_bytes",uint64_t(vm.wire_count)*page},
    {"system_compressor_bytes",uint64_t(vm.compressor_page_count)*page},{"system_file_backed_bytes",uint64_t(vm.external_page_count)*page}};
  if(initialized.load(std::memory_order_acquire)) {
    result["mlx_active_bytes"]=mlx::core::get_active_memory(); result["mlx_cache_bytes"]=mlx::core::get_cache_memory(); result["mlx_peak_bytes"]=mlx::core::get_peak_memory();
    @autoreleasepool { id<MTLDevice> device=MTLCreateSystemDefaultDevice(); if(!device) throw Error("RuntimeError","Metal device unavailable during memory sampling"); result["metal_process_allocated_bytes"]=device.currentAllocatedSize; }
  }
  return result;
}

struct ResourceMonitor::Impl {
  bool require_ac,closed=false,stop=false,owns=false;
  std::optional<fs::path> report_path;
  Json metadata,samples=Json::array(),power_start=nullptr,power_end=nullptr,sampled_power=nullptr,exception=nullptr;
  std::function<void(const Json&)> callback;
  mutable std::mutex mutex;
  std::mutex wait_mutex; std::condition_variable wake; std::thread thread;
  std::exception_ptr error;
  int log_fd=-1;
  double start=monotonic_seconds(),next_power=5;
  Impl(bool ac,std::optional<fs::path> report,Json meta,std::function<void(const Json&)> cb):require_ac(ac),report_path(std::move(report)),metadata(std::move(meta)),callback(std::move(cb)) {}
  ~Impl() { if(log_fd>=0) ::close(log_fd); }
  void latch(std::exception_ptr e) { std::lock_guard lock(mutex); if(!error) error=e; }
  void sample() {
    Json sample=memory_snapshot(); sample["elapsed_seconds"]=monotonic_seconds()-start;
    if(require_ac) {
      if(sample["elapsed_seconds"].get<double>()>=next_power) { sampled_power=power_source(); next_power=sample["elapsed_seconds"].get<double>()+5; }
      sample["ac_connected"]=sampled_power.at("ac_connected");
    }
    { std::lock_guard lock(mutex); samples.push_back(sample); }
    if(log_fd>=0) write_fd(log_fd,sample.dump()+"\n");
    if(require_ac&&!sample.at("ac_connected").get<bool>()) throw Error("RuntimeError","AC power disconnected during acceptance execution");
    if(callback) callback(sample);
  }
  Json report() const {
    std::lock_guard lock(mutex); Json maxima=Json::object();
    for(const auto& sample:samples) for(auto it=sample.begin();it!=sample.end();++it) if(it.key().ends_with("_bytes")) maxima[it.key()]=std::max(maxima.value(it.key(),uint64_t(0)),it.value().get<uint64_t>());
    return {{"memory_counter_schema",2},{"swap_counter_source","host_statistics64.swapins/swapouts"},{"sampling_interval_seconds",.25},
      {"power_sampling_interval_seconds",require_ac?Json(5.):Json(nullptr)},{"metadata",metadata},{"power_start",power_start},{"power_end",power_end},
      {"start",samples.empty()?Json(nullptr):samples.front()},{"end",samples.empty()?Json(nullptr):samples.back()},
      {"sampled_maxima",maxima},{"monitor_error",error?Json(error_text(error)):Json(nullptr)},{"exception",exception},{"samples",samples}};
  }
};
ResourceMonitor::ResourceMonitor(bool ac,const std::optional<fs::path>& log,const std::optional<fs::path>& report,Json metadata,std::function<void(const Json&)> callback)
  :impl_(std::make_unique<Impl>(ac,report,std::move(metadata),std::move(callback))) {
  auto& p=*impl_;
  try {
    for(const auto& path:{log,report}) if(path && fs::exists(*path)) throw Error("FileExistsError","Resource evidence already exists: "+path->string());
    if(log) p.log_fd=exclusive_file(*log); p.owns=true;
    p.power_start=power_source(); p.sampled_power=p.power_start;
    if(ac&&!p.power_start.at("ac_connected").get<bool>()) throw Error("RuntimeError","AC power is required for an acceptance benchmark");
    p.sample();
    p.thread=std::thread([&p]{std::unique_lock lock(p.wait_mutex); while(!p.wake.wait_for(lock,std::chrono::milliseconds(250),[&p]{return p.stop;})) { lock.unlock(); try { p.sample(); } catch(...) { p.latch(std::current_exception()); } lock.lock(); }});
  } catch(...) { auto error=std::current_exception(); p.latch(error); p.exception=error_text(error); try {close();} catch(...) {} std::rethrow_exception(error); }
}
ResourceMonitor::~ResourceMonitor() { if(!impl_->closed) try { close(); } catch(...) { diagnostic(error_text(std::current_exception())); } }
void ResourceMonitor::check() const { check_cancelled(); std::exception_ptr error; {std::lock_guard lock(impl_->mutex);error=impl_->error;} if(error) std::rethrow_exception(error); }
Json ResourceMonitor::report() const { return impl_->report(); }
void ResourceMonitor::record_exception(std::exception_ptr error) {std::lock_guard lock(impl_->mutex);impl_->exception=error?Json(error_text(error)):Json(nullptr);}
void ResourceMonitor::close() {
  auto& p=*impl_; if(p.closed) { check(); return; } p.closed=true;
  {std::lock_guard lock(p.wait_mutex);p.stop=true;} p.wake.notify_all(); if(p.thread.joinable()) p.thread.join();
  if(std::uncaught_exceptions()) {std::lock_guard lock(p.mutex);if(p.exception.is_null())p.exception="Native exception unwinding";}
  if(p.owns) {
    try { p.sample(); } catch(...) {p.latch(std::current_exception());}
    try { auto power=power_source(); { std::lock_guard lock(p.mutex);p.power_end=power; } if(p.require_ac&&!power.at("ac_connected").get<bool>()) throw Error("RuntimeError","AC power disconnected during acceptance benchmark"); } catch(...) {p.latch(std::current_exception());}
  }
  if(p.log_fd>=0) {int fd=p.log_fd;p.log_fd=-1;if(::close(fd)) p.latch(std::make_exception_ptr(Error("OSError","Closing resource log failed")));}
  if(p.report_path && p.owns) {int fd=-1;try {fd=exclusive_file(*p.report_path);write_fd(fd,p.report().dump(2)+"\n");if(::close(fd)) {fd=-1;os_error("Closing resource report");}fd=-1;}catch(...){if(fd>=0)::close(fd);p.latch(std::current_exception());}}
  check();
}

struct GPUExecution::Impl {
  std::unique_lock<std::recursive_mutex> lock{gpu_mutex,std::try_to_lock};
  std::unique_ptr<ResourceMonitor> monitor;
  Json baseline=nullptr;
  double budget; bool registered=false,closed=false,precision_verified=false;
  explicit Impl(double b):budget(b) {}
  void sample(const Json& sample) {
    if(baseline.is_null()) baseline=sample;
    if(sample.at("physical_footprint_bytes").get<double>()>budget*GiB) throw Error("MemoryError","Process footprint exceeds "+std::to_string(budget)+" GiB budget");
    int pressure=sample.at("system_memory_pressure_level"); if(pressure!=1) throw Error("MemoryError","System memory pressure is not normal (level="+std::to_string(pressure)+")");
    if(sample.at("system_available_bytes").get<uint64_t>()<2*GiB) throw Error("MemoryError","Less than 2 GiB of available system memory remains");
    double swapped=sample.at("system_swap_out_bytes").get<double>()-baseline.at("system_swap_out_bytes").get<double>();
    double growth=sample.at("system_swap_used_bytes").get<double>()-baseline.at("system_swap_used_bytes").get<double>();
    if(swapped>64*MiB||growth>128*MiB) throw Error("MemoryError","Stopping GPU workload after new swapping: "+std::to_string(swapped/MiB)+" MiB out, "+std::to_string(growth/MiB)+" MiB used growth");
  }
};
GPUExecution::GPUExecution(double budget,bool ac):impl_(std::make_unique<Impl>(budget)) {
  auto& p=*impl_;
  if(!p.lock.owns_lock()) throw Error("RuntimeError","Another thread owns Lyra GPU execution");
  if(!std::isfinite(budget)||budget<=5||budget>sysctl_value<uint64_t>("hw.memsize")/double(GiB)-4) throw Error("ValueError","Memory budget must exceed 5 GiB and leave 4 GiB OS headroom");
  try {
    if(gpu_owners.empty()) {
      fs::path path=fs::temp_directory_path()/("lyra-gpu-"+std::to_string(getuid())+".lock");
      gpu_fd=open(path.c_str(),O_CREAT|O_RDWR|O_CLOEXEC|O_NOFOLLOW,0600); if(gpu_fd<0) os_error("Opening GPU lock");
      struct stat st{}; if(fstat(gpu_fd,&st)||!S_ISREG(st.st_mode)||st.st_uid!=getuid()||(st.st_mode&077)!=0||st.st_nlink!=1) {::close(gpu_fd);gpu_fd=-1;throw Error("RuntimeError","Unsafe Lyra GPU lock file");}
      if(flock(gpu_fd,LOCK_EX|LOCK_NB)) {::close(gpu_fd);gpu_fd=-1;throw Error("RuntimeError","Another Lyra process owns the GPU; run workloads serially");}
    }
    gpu_owners.emplace_back(&p,budget); p.registered=true;
    initialize_runtime();
    double effective=budget;for(const auto& owner:gpu_owners) effective=std::min(effective,owner.second);
    mlx::core::set_memory_limit(size_t((effective-5)*GiB));mlx::core::set_cache_limit(128*MiB);
    Json metadata={{"gpu_backend","mlx"},{"memory_budget_gib",budget},{"whole_process_limit_kind","sampled"},{"maximum_new_swap_out_bytes",64*MiB},
      {"mlx_default_device","gpu"},{"mlx_advisory_memory_limit_bytes",size_t((effective-5)*GiB)},{"mlx_cache_limit_bytes",128*MiB},{"mlx_tf32_enabled",false},
      {"metal_allocation_limit_bytes",size_t((effective-1)*GiB)},{"metal_allocation_limit_kind","native decoder buffers and graph outputs; graph intermediates additionally sampled"}};
    p.monitor=std::make_unique<ResourceMonitor>(ac,std::nullopt,std::nullopt,metadata,[&p](const Json& s){p.sample(s);});
    gpu_monitors.push_back(p.monitor.get());
    static bool verified=false;
    if(!verified) {auto operand=mlx::core::full({128,128},1.+std::ldexp(1.,-14),mlx::core::float32);auto difference=mlx::core::max(mlx::core::abs(mlx::core::subtract(mlx::core::matmul(operand,mlx::core::eye(128,mlx::core::float32)),operand)));if(difference.item<float>()!=0) throw Error("RuntimeError","MLX was initialized with reduced FP32 precision; restart with MLX_ENABLE_TF32=0 before any MLX computation");verified=true;}
    p.precision_verified=true;
    check();
  } catch(...) {auto error=std::current_exception();record_exception(error);try {close();}catch(...){}std::rethrow_exception(error);}
}
GPUExecution::~GPUExecution() {try {close();}catch(...){diagnostic(error_text(std::current_exception()));}}
void GPUExecution::check() const {check_cancelled();std::unique_lock lock(gpu_mutex,std::try_to_lock);if(!lock.owns_lock())throw Error("RuntimeError","Another thread owns Lyra GPU execution");if(impl_->closed) throw Error("RuntimeError","GPU execution context is closed");for(auto* monitor:gpu_monitors) monitor->check();}
Json GPUExecution::report() const {
  Json report=impl_->monitor?impl_->monitor->report():Json::object();
  report["gpu_backend"]="mlx";report["memory_budget_gib"]=impl_->budget;report["maximum_new_swap_out_bytes"]=64*MiB;
  report["metadata"]["mlx_fp32_precision_verified"]=impl_->precision_verified;
  return report;
}
void GPUExecution::record_exception(std::exception_ptr error) {if(impl_->monitor)impl_->monitor->record_exception(error);}
void GPUExecution::close() {
  auto& p=*impl_;if(p.closed)return;p.closed=true;std::exception_ptr error;
  if(p.monitor) {try {p.monitor->close();}catch(...){error=std::current_exception();}std::erase(gpu_monitors,p.monitor.get());}
  if(p.registered) {std::erase_if(gpu_owners,[&p](const auto& owner){return owner.first==&p;});p.registered=false;}
  if(gpu_owners.empty()&&gpu_fd>=0){int fd=gpu_fd;gpu_fd=-1;if(::close(fd)&&!error)error=std::make_exception_ptr(Error("OSError","Closing GPU lock failed"));}
  if(p.lock.owns_lock())p.lock.unlock();if(error)std::rethrow_exception(error);
}

struct Progress::Impl {
  bool enabled,finished=false,tty=false,stop=false; int columns=80,width=0,complete=0,total,exceptions=std::uncaught_exceptions();
  std::string label,unit;double start=monotonic_seconds(),last=0,interval=5;
  std::mutex mutex;std::condition_variable wake;std::thread thread;
  Impl(bool e,std::string l,std::string u,int t):enabled(e),total(t),label(ascii(std::move(l))),unit(ascii(std::move(u))) {}
  void render(const char* status=nullptr,bool force=false) {
    if(!enabled)return;double now=monotonic_seconds();if(!force&&now-last<interval)return;last=now;double elapsed=std::max(0.,now-start);
    std::ostringstream suffix;suffix<<std::fixed<<std::setprecision(1);
    if(total>0) {if(tty){int fill=std::clamp(int(8.*complete/total),0,8);suffix<<"["<<std::string(fill,'#')<<std::string(8-fill,'-')<<"] ";}suffix<<complete<<"/"<<total<<" "<<(unit.empty()?"items":unit)<<" ("<<std::setprecision(0)<<100.*complete/total<<"%) | "<<std::setprecision(1);}
    else if(!unit.empty()||complete) suffix<<complete<<" "<<(unit.empty()?"items":unit)<<" | ";
    if(unit=="tokens")suffix<<(elapsed>0?complete/elapsed:0)<<" tokens/s | ";suffix<<(tty?"":"elapsed ")<<elapsed<<"s";
    std::string prefix=status?status:(tty?std::string(1,"|/-\\"[int(elapsed/.25)%4]):(force&&complete==0?"Starting":"Running"));
    std::string shown=label;if(tty){int available=columns-1-int(std::string("[YuE2] "+prefix+" : "+suffix.str()).size());if(int(shown.size())>std::max(0,available))shown=shown.substr(0,std::max(0,available-3))+(available>=3?"...":"");}
    std::string text="[YuE2] "+prefix+" "+shown+": "+suffix.str();
    std::lock_guard output(stderr_mutex);
    if(tty){text.resize(std::min(text.size(),size_t(columns-1)));std::string padding(std::max(0,width-int(text.size())),' ');if(std::fprintf(stderr,"\r%s%s%s",text.c_str(),padding.c_str(),status?"\n":"")<0)enabled=false;width=status?0:int(text.size());}
    else if(std::fprintf(stderr,"%s\n",text.c_str())<0)enabled=false;
    if(std::fflush(stderr))enabled=false;
  }
  void finish(const char* status) { {std::lock_guard lock(mutex);if(finished)return;finished=true;stop=true;render(status,true);}wake.notify_all();if(thread.joinable())thread.join(); }
};
Progress::Progress(bool enabled,std::string label,std::string unit,int total):impl_(std::make_unique<Impl>(enabled,std::move(label),std::move(unit),total)) {
  if(total<0)throw Error("ValueError","total must be nonnegative");auto& p=*impl_;if(!enabled)return;
  p.tty=isatty(STDERR_FILENO);if(p.tty){struct winsize ws{};const char* cols=std::getenv("COLUMNS");char* end=nullptr;long value=cols?std::strtol(cols,&end,10):0;if(cols&&end!=cols&&!*end&&value>0&&value<=100000)p.columns=int(value);else if(!ioctl(STDERR_FILENO,TIOCGWINSZ,&ws)&&ws.ws_col)p.columns=ws.ws_col;p.tty=p.columns>=60;}
  p.interval=p.tty?.25:5.;p.render(nullptr,true);p.thread=std::thread([&p]{std::unique_lock lock(p.mutex);while(!p.wake.wait_for(lock,std::chrono::duration<double>(p.interval),[&p]{return p.stop;}))p.render();});
}
Progress::~Progress(){impl_->finish(cancellation_requested()?"Cancelled":(std::uncaught_exceptions()>impl_->exceptions?"Failed":"Completed"));}
void Progress::update(int complete,int total){if(complete<0||total<0)throw Error("ValueError","Progress counts must be nonnegative");auto& p=*impl_;std::lock_guard lock(p.mutex);if(p.finished)return;p.complete=complete;if(total>0)p.total=total;p.render();}
void Progress::advance(){auto& p=*impl_;std::lock_guard lock(p.mutex);if(p.finished)return;++p.complete;p.render();}
void Progress::finish(bool truncated){impl_->finish(truncated?(impl_->tty?"Limit reached":"Finished (generation limit reached)"):"Completed");}
} // namespace lyra
