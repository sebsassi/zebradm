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
cmake_minimum_required(VERSION 3.19)
project(zebradm 
    VERSION 0.1.0 
    LANGUAGES CXX
)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
endif()

add_library(zebradm
    src/zebra_angle_integrator_core.cpp src/zebra_angle_integrator.cpp src/zebra_parallel_angle_integrator.cpp src/legendre.cpp src/affine_legendre.cpp
    src/affine_legendre_integral.cpp src/radon_integrator.cpp src/radon_util.cpp src/zebra_radon.cpp src/zernike_recursions.cpp
)

target_include_directories(zebradm
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(zebradm PUBLIC cxx_std_23)

include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION include/zebradm)

install(TARGETS zebradm EXPORT zebradm-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "cmake/config.cmake.in"
    "zebradm/zebradm-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/zebradm
    PATH_VARS CMAKE_INSTALL_LIBDIR
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/zebradm/zebradm-config-version.cmake"
    VERSION ${CMAKE_PROJECT_VERSION}
    COMPATIBILITY ExactVersion
)

install(EXPORT zebradm-targets
    FILE zebradm-targets.cmake
    NAMESPACE zebradm::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/zebradm
)

install(
    FILES 
        "${CMAKE_CURRENT_BINARY_DIR}/zebradm/zebradm-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/zebradm/zebradm-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/zebradm
    COMPONENT Devel
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(zebradm PUBLIC OpenMP::OpenMP_CXX)
endif()

find_package(cubage CONFIG)
if((NOT cubage_FOUND) AND (BUILD_BENCHMARKS OR BUILD_TESTING))
    message(FATAL_ERROR "cubage not found: benchmarks or tests cannot be built without cubage")
else()
    target_link_libraries(zebradm PUBLIC cubage::cubage)
endif()

find_package(zest CONFIG REQUIRED)
target_link_libraries(zebradm PUBLIC zest::zest)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_DOCS)
    add_subdirectory(docs)
endif()
