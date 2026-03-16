# Root CMakeLists.txt for delta_analysis project
cmake_minimum_required(VERSION 3.15)   # Require CMake 3.15+ for proper FetchContent

# 1. MSVC debug information format policy (CMP0141)
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "ProgramDatabase")
endif()

project("delta_analysis" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. OpenMP configuration (LLVM backend for MSVC)
if(MSVC)
    # Use the new /openmp:llvm flag (requires MSVC 19.29+ and LLVM tools)
    set(OpenMP_CXX_FLAGS "/openmp:llvm")
    
    if(NOT TARGET OpenMP::OpenMP_CXX)
        # Manually create an imported target for OpenMP with LLVM backend
        add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
        target_compile_options(OpenMP::OpenMP_CXX INTERFACE /openmp:llvm)
        # Link against libomp.lib; if linker error LNK1104 occurs,
        # add the LLVM library path via link_directories().
        target_link_libraries(OpenMP::OpenMP_CXX INTERFACE libomp.lib)
        target_compile_definitions(OpenMP::OpenMP_CXX INTERFACE _OPENMP_NO_LLVM_HELPER)
    endif()
else()
    # For non-MSVC compilers, use the standard FindOpenMP module.
    find_package(OpenMP REQUIRED)
endif()


# 3. External dependencies (centralised)
find_package(Boost REQUIRED)          # Boost libraries (headers mostly)
find_package(Eigen3 REQUIRED)         # Eigen3 for linear algebra
find_package(fmt CONFIG REQUIRED)     # fmtlib for formatting

# 4. Google Test (included once for the whole project)
include(FetchContent)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)   # Ensure consistent CRT with MSVC
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)           # Do not install GTest

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2               # Use a specific release tag
  GIT_SHALLOW    TRUE                   # Shallow clone to save bandwidth/time
)
FetchContent_MakeAvailable(googletest)

# 5. Core interface library (delta_core)
add_library(delta_core INTERFACE)
target_include_directories(delta_core INTERFACE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(delta_core INTERFACE 
    Boost::boost 
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    fmt::fmt
)

if(MSVC)
    # MSVC‑specific compile options: enable C++ exceptions, set __cplusplus correctly.
    target_compile_options(delta_core INTERFACE /EHsc /Zc:__cplusplus)
    # Define EIGEN_HAS_OPENMP to let Eigen use OpenMP.
    target_compile_definitions(delta_core INTERFACE EIGEN_HAS_OPENMP)
endif()

# --- Rational backend selection ---
# Choose the arithmetic mode for delta::Rational.
# Options: NATIVE_DOUBLE  - use built‑in double (fast, limited precision)
#          BIN_FLOAT      - binary floating‑point (cpp_bin_float, default 100 digits)
#          FIXED_RATIONAL - fixed‑bit rational (rational_adaptor with fixed integer, default 256 bits)
#          EXACT_RATIONAL - exact unlimited rational (rational_adaptor with dynamic integers)
set(DELTA_RATIONAL_MODE "EXACT_RATIONAL" CACHE STRING "Rational arithmetic mode")
set_property(CACHE DELTA_RATIONAL_MODE PROPERTY STRINGS NATIVE_DOUBLE BIN_FLOAT FIXED_RATIONAL EXACT_RATIONAL)

if(DELTA_RATIONAL_MODE STREQUAL "NATIVE_DOUBLE")
    target_compile_definitions(delta_core INTERFACE DELTA_RATIONAL_MODE_NATIVE_DOUBLE)

elseif(DELTA_RATIONAL_MODE STREQUAL "BIN_FLOAT")
    target_compile_definitions(delta_core INTERFACE DELTA_RATIONAL_MODE_BIN_FLOAT)
    # Optional: number of decimal digits (default 100)
    set(DELTA_BIN_FLOAT_DIGITS "100" CACHE STRING "Number of decimal digits for binary float (cpp_bin_float)")
    target_compile_definitions(delta_core INTERFACE DELTA_BIN_FLOAT_DIGITS=${DELTA_BIN_FLOAT_DIGITS})

elseif(DELTA_RATIONAL_MODE STREQUAL "FIXED_RATIONAL")
    target_compile_definitions(delta_core INTERFACE DELTA_RATIONAL_MODE_FIXED_RATIONAL)
    # Optional: bit width for the fixed‑size integer (default 256)
    set(DELTA_RATIONAL_BITS "256" CACHE STRING "Bit width for fixed‑bit rational (FIXED_RATIONAL mode)")
    target_compile_definitions(delta_core INTERFACE DELTA_RATIONAL_BITS=${DELTA_RATIONAL_BITS})

else()   # EXACT_RATIONAL (default)
    target_compile_definitions(delta_core INTERFACE DELTA_RATIONAL_MODE_EXACT_RATIONAL)
endif()
# --------------

# Helper variable: directory containing the MSVC compiler (used for DLL discovery in tests)
get_filename_component(MSVC_BIN_DIR ${CMAKE_CXX_COMPILER} DIRECTORY CACHE INTERNAL "")

# Add subdirectories for tests, examples, and benchmarks
#add_subdirectory("tests/basic")
#add_subdirectory("tests/calculus")
#add_subdirectory("tests/regulative_ideas")
add_subdirectory("tests/geometry")
#add_subdirectory("tests/numerical")
#add_subdirectory("tests/solvers")
#add_subdirectory("examples/arbitrary_regulative_ideas")
#add_subdirectory("benchmarks")