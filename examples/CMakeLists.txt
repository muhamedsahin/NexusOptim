add_executable(basic_sgd_example basic_sgd_example.cpp)
target_link_libraries(basic_sgd_example PRIVATE nexus_optim::nexus_optim)

add_executable(train_loop_with_nexus_optim train_loop_with_nexus_optim.cpp)
target_link_libraries(train_loop_with_nexus_optim PRIVATE nexus_optim::nexus_optim)

if(MSVC)
  target_compile_options(basic_sgd_example PRIVATE /W4)
  target_compile_options(train_loop_with_nexus_optim PRIVATE /W4)
endif()
