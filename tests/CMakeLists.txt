macro(create_test TESTNAME)
    add_executable(${TESTNAME}.test ${TESTNAME}.cpp)
    target_include_directories(${TESTNAME}.test
        PRIVATE ${PROJECT_SOURCE_DIR}/include
    )
    target_compile_options(${TESTNAME}.test
        PRIVATE
            $<$<CONFIG:Debug>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Werror;-g;-O0>
    )
    target_link_libraries(${TESTNAME}.test fastd3m)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME}.test)
endmacro()

create_test(test_radon_integrator)
create_test(test_legendre_recursion)
create_test(test_affine_legendre)
create_test(test_radon_util)
create_test(test_radon_transformer)
create_test(test_mutual_convergence)