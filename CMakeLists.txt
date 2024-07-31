cmake_minimum_required(VERSION 3.12)
project(zebradm 
    VERSION 0.1.0 
    LANGUAGES CXX
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(zebradm
    src/zebra_angle_integrator_core.cpp src/zebra_angle_integrator.cpp src/zebra_parallel_angle_integrator.cpp src/legendre.cpp src/affine_legendre.cpp
    src/affine_legendre_integral.cpp src/radon_integrator.cpp src/radon_util.cpp
)

add_subdirectory(external/cubage)

target_include_directories(zebradm
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_options(zebradm
    PRIVATE
        $<$<CONFIG:Debug>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Werror;-g;-O0>
        $<$<CONFIG:Release>:-Wall;-Wextra;-Wpedantic;-Wconversion;-g;-O3>
)

target_compile_features(zebradm PUBLIC cxx_std_20)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(zebradm PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(zebradm PUBLIC cubage zest)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()