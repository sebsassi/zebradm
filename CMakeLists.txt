# Copyright (c) 2024 Sebastian Sassi

# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal # in the Software without restriction, including without limitation the rights # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.
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
    src/affine_legendre_integral.cpp src/radon_integrator.cpp src/radon_util.cpp src/zebra_radon.cpp src/zernike_recursions.cpp
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
        $<$<CONFIG:Debug>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Werror;-g;-O3>
        $<$<CONFIG:Release>:-Wall;-Wextra;-Wpedantic;-Wconversion;-g;-O3>
)

target_compile_features(zebradm PUBLIC cxx_std_20)

install(DIRECTORY include/ DESTINATION include/zebradm)

install(TARGETS zebradm
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

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