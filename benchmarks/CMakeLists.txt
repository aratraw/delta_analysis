# benchmarks/CMakeLists.txt
# Configuration for performance benchmarks of core Δ‑analysis components.
# Benchmarks use the Google Benchmark library and measure execution time
# of key operations: path advancement, operational function extension,
# Riemann sum computation, and adaptive path refinement.

# Locate the Google Benchmark package (required).
find_package(benchmark REQUIRED)

# =============================================================================
# Единый бинарник, объединяющий все четыре бенчмарк-сюиты
# =============================================================================

add_executable(benchmark_core benchmark_core.cpp)
target_link_libraries(benchmark_core PRIVATE benchmark::benchmark delta_core)
target_include_directories(benchmark_core PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_options(benchmark_core PRIVATE /Zc:__cplusplus)