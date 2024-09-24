macro(create_test TESTNAME)
    add_executable(${TESTNAME}.test ${TESTNAME}.cpp)
    target_include_directories(${TESTNAME}.test
        PRIVATE ${PROJECT_SOURCE_DIR}/include
    )
    target_compile_options(${TESTNAME}.test
        PRIVATE
            $<$<CONFIG:Debug>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Werror;-g;-O0>
    )
    target_link_libraries(${TESTNAME}.test zebradm)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME}.test)
endmacro()

macro(create_experiment EXPNAME)
    add_executable(${EXPNAME}.exp ${EXPNAME}.cpp)
    target_include_directories(${EXPNAME}.exp
        PRIVATE ${PROJECT_SOURCE_DIR}/include
    )
    target_compile_options(${EXPNAME}.exp
        PRIVATE
            $<$<CONFIG:Debug>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Werror;-g;-O0>
    )
    target_link_libraries(${EXPNAME}.exp zebradm)
endmacro()

create_test(test_radon_angle_integrator)
create_test(test_legendre_recursion)
create_test(test_affine_legendre)
create_test(test_affine_legendre_integral)
create_test(test_radon_util)
create_test(test_zebra_isotropic_angle_integrator)
create_test(test_zebra_isotropic_transverse_angle_integrator)
create_test(test_mutual_convergence)
create_test(test_zernike_recursions)

create_experiment(exp_affine_legendre_integral)