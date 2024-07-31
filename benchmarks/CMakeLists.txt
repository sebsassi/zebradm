macro(create_nanobench BENCHNAME)
    add_executable(${BENCHNAME}.bench
        ${BENCHNAME}.cpp nanobench.cpp
    )
    target_include_directories(${BENCHNAME}.bench
        PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_options(${BENCHNAME}.bench
        PRIVATE
            $<$<CONFIG:Debug>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Werror;-g;-O3;-mavx2;-mfma>
    )
    target_link_libraries(${BENCHNAME}.bench zebradm)
endmacro()

create_nanobench(benchmark_radon_transformer_without_response)