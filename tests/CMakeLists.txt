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
macro(create_test TESTNAME)
    add_executable(${TESTNAME}.test ${TESTNAME}.cpp)
    target_include_directories(${TESTNAME}.test
        PRIVATE ${PROJECT_SOURCE_DIR}/include
    )
    target_link_libraries(${TESTNAME}.test zebradm)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME}.test)
endmacro()

macro(create_experiment EXPNAME)
    add_executable(${EXPNAME}.exp ${EXPNAME}.cpp)
    target_include_directories(${EXPNAME}.exp
        PRIVATE ${PROJECT_SOURCE_DIR}/include
    )
    target_link_libraries(${EXPNAME}.exp zebradm)
endmacro()

create_test(test_linalg)
create_test(test_time)
create_test(test_astro)
create_test(test_radon_angle_integrator)
create_test(test_legendre_recursion)
create_test(test_affine_legendre)
create_test(test_affine_legendre_integral)
create_test(test_zonal_glq_transformer)
create_test(test_zebra_radon)
create_test(test_zebra_isotropic_angle_integrator)
create_test(test_zebra_anisotropic_angle_integrator)
create_test(test_zebra_isotropic_transverse_angle_integrator)
create_test(test_mutual_convergence)
create_test(test_zernike_recursions)

create_experiment(exp_affine_legendre_integral)
create_experiment(exp_zebra_convergence)
