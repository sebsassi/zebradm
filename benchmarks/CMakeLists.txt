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

create_other_bench(zebra_isotropic_angle_integrator_error)
create_other_bench(zebra_isotropic_angle_integrator_transverse_error)
create_other_bench(zebra_anisotropic_angle_integrator_error)
create_other_bench(zebra_anisotropic_angle_integrator_transverse_error)