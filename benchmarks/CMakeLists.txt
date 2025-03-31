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
macro(create_nanobench BENCHNAME)
    add_executable(${BENCHNAME}.bench
        ${BENCHNAME}.cpp nanobench.cpp
    )
    target_include_directories(${BENCHNAME}.bench
        PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_options(${BENCHNAME}.bench
        PRIVATE
            $<$<CONFIG:Release>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Werror;-g;-O3;-mavx2;-mfma>
    )
    target_link_libraries(${BENCHNAME}.bench zebradm)
endmacro()

create_nanobench(benchmark_zebra_isotropic_angle_integrator)
create_nanobench(benchmark_zebra_isotropic_angle_integrator_transverse)
create_nanobench(benchmark_zebra_anisotropic_angle_integrator)
create_nanobench(benchmark_zebra_anisotropic_angle_integrator_transverse)
create_nanobench(benchmark_zebra_parallel_isotropic_angle_integrator)
create_nanobench(benchmark_radon_isotropic_angle_integrator)
create_nanobench(benchmark_radon_isotropic_angle_integrator_transverse)
create_nanobench(benchmark_radon_anisotropic_angle_integrator)
create_nanobench(benchmark_radon_anisotropic_angle_integrator_transverse)
create_nanobench(benchmark_dist_grid_creation)
create_nanobench(benchmark_resp_grid_creation)

macro(create_other_bench BENCHNAME)
    add_executable(${BENCHNAME}.out
        ${BENCHNAME}.cpp
    )
    target_include_directories(${BENCHNAME}.out
        PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_options(${BENCHNAME}.out
        PRIVATE
            $<$<CONFIG:Release>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Werror;-g;-O3;-mavx2;-mfma>
    )
    target_link_libraries(${BENCHNAME}.out zebradm)
endmacro()

create_other_bench(radon_anisotropic_angle_integrator_error)
create_other_bench(radon_anisotropic_angle_integrator_transverse_error)
create_other_bench(zebra_isotropic_angle_integrator_error)
create_other_bench(zebra_isotropic_angle_integrator_transverse_error)
create_other_bench(zebra_anisotropic_angle_integrator_error)
create_other_bench(zebra_anisotropic_angle_integrator_transverse_error)