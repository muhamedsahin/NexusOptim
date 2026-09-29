include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(NEXUS_OPTIM_TEST_SOURCES
  test_sgd.cpp
  test_adam.cpp
  test_adamw.cpp
  test_algorithms.cpp
  test_lr_schedulers.cpp
  test_gradient_clipping.cpp
  test_against_pytorch_reference.cpp
)

add_executable(nexus_optim_tests ${NEXUS_OPTIM_TEST_SOURCES})
target_link_libraries(nexus_optim_tests PRIVATE nexus_optim::nexus_optim GTest::gtest_main)
string(REPLACE "\\" "/" NEXUS_OPTIM_SOURCE_DIR_SAFE "${CMAKE_SOURCE_DIR}")
target_compile_definitions(nexus_optim_tests PRIVATE
  NEXUS_OPTIM_SOURCE_DIR="${NEXUS_OPTIM_SOURCE_DIR_SAFE}")
if(MSVC)
  target_compile_options(nexus_optim_tests PRIVATE /W4)
endif()

if(NEXUS_OPTIM_WITH_CUDA)
  target_sources(nexus_optim_tests PRIVATE test_cuda.cpp)
  target_compile_definitions(nexus_optim_tests PRIVATE NEXUS_OPTIM_WITH_CUDA=1)
endif()

include(GoogleTest)
gtest_discover_tests(nexus_optim_tests)
