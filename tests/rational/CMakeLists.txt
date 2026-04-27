# tests/rational/CMakeLists.txt
add_executable(delta_tests_rational
    main_tests_rational.cpp
    rational_test.cpp
    rational_test_2.cpp
    interval_test.cpp
    batch_test.cpp
  lazy_rational_contract_tests.cpp
    lazy_test.cpp
    lazy_simplification_tests.cpp
    gc_test.cpp
    gc_reset_pool_edge_cases_tests.cpp
   pow_test.cpp
   transcendentals_correctness.cpp
    transcendentals_canonicalization_benchmark.cpp
   transcendentals_comparative.cpp
   performance_compare_test.cpp
#    performance_test.cpp
 )

target_include_directories(delta_tests_rational PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(delta_tests_rational
    PRIVATE
    delta_core
    gtest_main
)

if(MSVC)
    target_compile_options(delta_tests_rational PRIVATE /EHsc)
endif()

target_compile_definitions(delta_tests_rational PRIVATE DELTA_TESTING)

include(GoogleTest)
gtest_discover_tests(delta_tests_rational
    PROPERTIES
    ENVIRONMENT "PATH=${MSVC_BIN_DIR};$ENV{PATH}"
)