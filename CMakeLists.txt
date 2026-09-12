cmake_minimum_required(VERSION 3.25)

# The CLI uses the same Metal 4 baseline as the existing runtime, so it can be
# exercised before upgrading the development machine to the macOS 27 app target.
set(CMAKE_OSX_DEPLOYMENT_TARGET "26.2" CACHE STRING "Minimum macOS deployment version")
project(lyra VERSION 0.2.0 LANGUAGES C CXX OBJCXX)
if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  message(FATAL_ERROR "Lyra requires an Apple Silicon Mac")
endif()
if(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 26.2)
  message(FATAL_ERROR "Lyra requires the macOS 26.2 or newer M5 Metal runtime")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
endif()

include(FetchContent)
include(GNUInstallDirs)
# MLX is a native dependency. Its Python bindings, examples, and tests are not
# part of either the build or the installed CLI.
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_STUBS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_GGUF OFF CACHE BOOL "" FORCE)
set(MLX_METAL_JIT OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(MLX_METAL_PATH "${CMAKE_BINARY_DIR}/lib" CACHE PATH "Compiled MLX Metal kernels" FORCE)
file(MAKE_DIRECTORY "${MLX_METAL_PATH}")
FetchContent_Declare(mlx URL https://github.com/ml-explore/mlx/archive/refs/tags/v0.32.2.tar.gz DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(mlx)

find_package(CURL REQUIRED)
find_package(SndFile 1.2 CONFIG REQUIRED HINTS /opt/homebrew/opt/libsndfile/lib/cmake/SndFile)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
find_library(IOKIT_FRAMEWORK IOKit REQUIRED)

add_library(lyra_engine STATIC
  native/src/protocol.mm
  native/src/storage.cpp
  native/src/runtime.mm
  native/src/conversion.cpp
  native/src/ar.cpp
  native/src/sampling.cpp
  native/src/nar.cpp
  native/src/nar_attention.cpp
  native/src/vae.mm
  native/src/audio.cpp
  native/src/noise.cpp
  native/src/artifacts.cpp
  native/src/pipeline.cpp
  native/src/workflow.cpp
  native/src/bridge.mm)
target_include_directories(lyra_engine PUBLIC native/include)
target_link_libraries(lyra_engine PUBLIC mlx nlohmann_json::nlohmann_json
  PRIVATE CURL::libcurl SndFile::sndfile ${FOUNDATION_FRAMEWORK}
  ${METAL_FRAMEWORK} ${MPS_FRAMEWORK} ${MPSGRAPH_FRAMEWORK} ${IOKIT_FRAMEWORK})
target_compile_options(lyra_engine PRIVATE
  $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
  -Wall -Wextra -Wpedantic -ffp-contract=off)
target_compile_definitions(lyra_engine PUBLIC
  LYRA_VERSION="${PROJECT_VERSION}" LYRA_MLX_VERSION="0.32.2")
# The identity is derived from compiled core archives, not the embedding app.
# Keep it outside the hashed archive to avoid a circular build dependency.
set(CORE_IDENTITY_SOURCE "${CMAKE_BINARY_DIR}/core_identity.cpp")
add_custom_command(OUTPUT "${CORE_IDENTITY_SOURCE}"
  BYPRODUCTS "${CORE_IDENTITY_SOURCE}.sha256"
  COMMAND "${CMAKE_COMMAND}"
    "-DCORE=$<TARGET_FILE:lyra_engine>"
    "-DMLX=$<TARGET_FILE:mlx>"
    "-DJACCL=$<TARGET_FILE:jaccl>"
    "-DOUTPUT=${CORE_IDENTITY_SOURCE}"
    -P "${CMAKE_SOURCE_DIR}/cmake/CoreIdentity.cmake"
  DEPENDS lyra_engine mlx jaccl cmake/CoreIdentity.cmake
  VERBATIM)
add_library(lyra_build_identity STATIC "${CORE_IDENTITY_SOURCE}")
add_library(lyra_core INTERFACE)
target_link_libraries(lyra_core INTERFACE lyra_engine lyra_build_identity)

add_executable(lyra native/src/main.mm native/src/cli.cpp native/src/terminal.cpp)
target_compile_options(lyra PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_link_libraries(lyra PRIVATE lyra_core)
set_target_properties(lyra PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  INSTALL_RPATH "@executable_path/../${CMAKE_INSTALL_LIBDIR}")
install(TARGETS lyra RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT lyra)
install(FILES "${MLX_METAL_PATH}/mlx.metallib"
  DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lyra)
install(FILES LICENSE vendor/yue/MODEL_LICENSE vendor/yue/THIRD_PARTY_NOTICES.md
  DESTINATION ${CMAKE_INSTALL_DATADIR}/lyra/licenses COMPONENT lyra)
install(DIRECTORY vendor/yue/licenses/ native/licenses/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/lyra/licenses COMPONENT lyra)
install(FILES "${mlx_SOURCE_DIR}/LICENSE"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/lyra/licenses RENAME MLX-MIT.txt COMPONENT lyra)

# A local Swift package consumes the same static core and pinned MLX build.
# No shared Lyra library is installed; only the CLI retains an install target.
find_program(XCRUN_EXECUTABLE xcrun REQUIRED)
add_custom_target(lyra_swift_package
  COMMAND "${CMAKE_COMMAND}"
    "-DXCRUN=${XCRUN_EXECUTABLE}"
    "-DCORE=$<TARGET_FILE:lyra_engine>"
    "-DIDENTITY=$<TARGET_FILE:lyra_build_identity>"
    "-DBUILD_ID=${CORE_IDENTITY_SOURCE}.sha256"
    "-DMLX=$<TARGET_FILE:mlx>"
    "-DJACCL=$<TARGET_FILE:jaccl>"
    "-DWORK=${CMAKE_BINARY_DIR}/swift-package"
    "-DPACKAGE=${CMAKE_SOURCE_DIR}/native/swift"
    "-DHEADERS=${CMAKE_SOURCE_DIR}/native/include/CLyraCore"
    "-DMETALLIB=${MLX_METAL_PATH}/mlx.metallib"
    "-DMLX_LICENSE=${mlx_SOURCE_DIR}/LICENSE"
    "-DROOT=${CMAKE_SOURCE_DIR}"
    -P "${CMAKE_SOURCE_DIR}/cmake/SwiftPackage.cmake"
  DEPENDS lyra_core mlx jaccl
  VERBATIM)
