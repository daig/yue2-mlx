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

find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(IOKIT_FRAMEWORK IOKit REQUIRED)

add_library(lyra_core STATIC
  native/src/protocol.mm
  native/src/storage.cpp
  native/src/runtime.mm)
target_include_directories(lyra_core PUBLIC native/include)
target_link_libraries(lyra_core PUBLIC mlx nlohmann_json::nlohmann_json
  PRIVATE ${FOUNDATION_FRAMEWORK} ${METAL_FRAMEWORK} ${IOKIT_FRAMEWORK})
target_compile_options(lyra_core PRIVATE
  $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
  -Wall -Wextra -Wpedantic -ffp-contract=off)
target_compile_definitions(lyra_core PUBLIC
  LYRA_VERSION="${PROJECT_VERSION}" LYRA_MLX_VERSION="0.32.2")
install(FILES LICENSE vendor/yue/MODEL_LICENSE vendor/yue/THIRD_PARTY_NOTICES.md
  DESTINATION ${CMAKE_INSTALL_DATADIR}/lyra/licenses)
install(DIRECTORY vendor/yue/licenses/ DESTINATION ${CMAKE_INSTALL_DATADIR}/lyra/licenses)
install(FILES "${mlx_SOURCE_DIR}/LICENSE" DESTINATION ${CMAKE_INSTALL_DATADIR}/lyra/licenses RENAME MLX-MIT.txt)
install(DIRECTORY native/include/lyra DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
