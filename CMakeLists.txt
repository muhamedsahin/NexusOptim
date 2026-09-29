cmake_minimum_required(VERSION 3.20)

project(NexusOptim
  VERSION 1.0.0
  DESCRIPTION "Fused, SIMD-accelerated parameter update library"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NEXUS_OPTIM_WITH_CUDA "Build fused CUDA update kernels" OFF)
option(NEXUS_OPTIM_BUILD_TESTS "Build GoogleTest suite" ON)
option(NEXUS_OPTIM_BUILD_BENCHMARKS "Build benchmarks" ON)
option(NEXUS_OPTIM_BUILD_EXAMPLES "Build examples" ON)
option(NEXUS_OPTIM_ENABLE_AVX512 "Compile AVX-512 kernels" ON)
option(NEXUS_OPTIM_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

if(NEXUS_OPTIM_WITH_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

set(NEXUS_OPTIM_SOURCES
  src/core/thread_pool.cpp
  src/core/optimizer_state.cpp
  src/simd/simd_dispatch.cpp
  src/simd/avx2_kernels.cpp
  src/simd/neon_kernels.cpp
  src/algorithms/instantiate.cpp
)

if(NEXUS_OPTIM_ENABLE_AVX512)
  list(APPEND NEXUS_OPTIM_SOURCES src/simd/avx512_kernels.cpp)
endif()

if(NEXUS_OPTIM_WITH_CUDA)
  list(APPEND NEXUS_OPTIM_SOURCES src/cuda/cuda_optimizer_kernels.cu)
endif()

add_library(nexus_optim STATIC ${NEXUS_OPTIM_SOURCES})
add_library(nexus_optim::nexus_optim ALIAS nexus_optim)

target_include_directories(nexus_optim PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(nexus_optim PUBLIC cxx_std_20)

if(NEXUS_OPTIM_ENABLE_AVX512)
  target_compile_definitions(nexus_optim PUBLIC NEXUS_OPTIM_ENABLE_AVX512=1)
endif()
if(NEXUS_OPTIM_WITH_CUDA)
  target_compile_definitions(nexus_optim PUBLIC NEXUS_OPTIM_WITH_CUDA=1)
  set_target_properties(nexus_optim PROPERTIES CUDA_STANDARD 17)
  target_compile_options(nexus_optim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
  target_link_libraries(nexus_optim PUBLIC CUDA::cudart)
endif()

if(MSVC)
  target_compile_options(nexus_optim PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
    $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
  if(NEXUS_OPTIM_WARNINGS_AS_ERRORS)
    target_compile_options(nexus_optim PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/WX>)
  endif()
  # Only the AVX translation units are allowed to emit AVX instructions.
  set_source_files_properties(src/simd/avx2_kernels.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  if(NEXUS_OPTIM_ENABLE_AVX512)
    set_source_files_properties(src/simd/avx512_kernels.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  endif()
else()
  target_compile_options(nexus_optim PRIVATE -Wall -Wextra -Wpedantic)
  if(NEXUS_OPTIM_WARNINGS_AS_ERRORS)
    target_compile_options(nexus_optim PRIVATE -Werror)
  endif()
  set_source_files_properties(src/simd/avx2_kernels.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  if(NEXUS_OPTIM_ENABLE_AVX512)
    set_source_files_properties(src/simd/avx512_kernels.cpp
      PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512dq;-mfma")
  endif()
  target_compile_options(nexus_optim PRIVATE $<$<CONFIG:Release>:-O3;-funroll-loops>)
endif()

# MatrixFlash-Pro is optional. Its Tensor is float-only; see adapters/matrixflash.hpp.
set(NEXUS_MATRIXFLASH_INCLUDE "" CACHE PATH "Path to MatrixFlash-Pro include/")
if(NEXUS_MATRIXFLASH_INCLUDE)
  target_include_directories(nexus_optim PUBLIC "${NEXUS_MATRIXFLASH_INCLUDE}")
endif()

include(CTest)

if(NEXUS_OPTIM_BUILD_TESTS)
  add_subdirectory(tests)
endif()
if(NEXUS_OPTIM_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
if(NEXUS_OPTIM_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

install(TARGETS nexus_optim ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
