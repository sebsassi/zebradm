/*
Copyright (c) 2024-2026 Sebastian Sassi

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*/

#include <cassert>

#include "units.hpp"
#include "vector.hpp"
#include "matrix.hpp"
#include "rotation.hpp"
#include "translation.hpp"

#include "mp-units/cartesian_vector.h"

bool test_vector_can_have_units()
{
    [[maybe_unused]] const auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    return true;
}

bool test_vector_with_units_can_add()
{
    const auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    [[maybe_unused]] const auto u = v + v;
    return true;
}

bool test_vector_with_units_can_subtract()
{
    const auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    [[maybe_unused]] const auto u = v - v;
    return true;
}

bool test_vector_with_units_can_multiply_by_scalar()
{
    const auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    [[maybe_unused]] const auto u = 3.0*v;
    return true;
}

bool test_vector_with_units_can_divide_by_scalar()
{
    const auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    [[maybe_unused]] const auto u = v/3.0;
    return true;
}

bool test_vector_with_units_can_add_assign()
{
    auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    v += v;
    return true;
}

bool test_vector_with_units_can_sub_assign()
{
    auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    v -= v;
    return true;
}

bool test_vector_with_units_can_mult_assign_by_scalar()
{
    auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    v *= 3.0;
    return true;
}

bool test_vector_with_units_can_div_assign_by_scalar()
{
    auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    v /= 3.0;
    return true;
}

bool test_vector_with_units_has_magnitude()
{
    const auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    [[maybe_unused]] const auto vm = zdm::mpu::magnitude(v);
    return true;
}

bool test_vector_with_units_can_swizzle()
{
    const auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    [[maybe_unused]] const auto u = zdm::la::swizzle<0, 1, 2, 1>(v);
    return true;
}

bool test_vector_with_units_has_dot_product()
{
    const auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    const auto u = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    [[maybe_unused]] const auto vu = zdm::la::dot(v, u);
    return true;
}

bool test_vector_with_units_can_be_normalized()
{
    const auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    [[maybe_unused]] const auto u = zdm::la::normalize(v);
    return true;
}

bool test_vector_with_units_has_matmul_with_matrix()
{
    const auto m = zdm::la::Matrix<double, 3, 3>::identity();
    const auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    [[maybe_unused]] const auto u = zdm::la::matmul(m, v);
    return true;
}

bool test_vector_with_units_has_matmul_with_rotation_matrix()
{
    const auto r = zdm::la::RotationMatrix<double, 3>::identity();
    const auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    [[maybe_unused]] const auto u = zdm::la::matmul(r, v);
    return true;
}

bool test_vector_with_units_can_be_multiplied_with_a_matrix()
{
    const auto m = zdm::la::Matrix<double, 3, 3>::identity();
    const auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    [[maybe_unused]] const auto u = m*v;
    return true;
}

bool test_vector_with_units_can_be_multiplied_with_a_rotation_matrix()
{
    const auto r = zdm::la::RotationMatrix<double, 3>::identity();
    const auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    [[maybe_unused]] const auto u = r*v;
    return true;
}

bool test_vector_with_units_has_quadratic_form_with_matrix()
{
    const auto m = zdm::la::Matrix<double, 3, 3>::identity();
    const auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    [[maybe_unused]] const auto q = zdm::la::quadratic_form(m, v);
    return true;
}

bool test_vector_with_units_has_quadratic_form_with_rotation_matrix()
{
    const auto r = zdm::la::RotationMatrix<double, 3>::identity();
    const auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    [[maybe_unused]] const auto q = zdm::la::quadratic_form(r, v);
    return true;
}

// NOTE: disabled until mp-units supports rank-2 tensor quantities
//
// bool test_matrix_can_have_units()
// {
//     [[maybe_unused]] const auto m = zdm::la::Matrix<double, 3, 3>::identity()*zdm::displacement[zdm::meter];
//     return true;
// }
//
// bool test_matrix_with_units_can_multiply()
// {
//     const auto m = zdm::la::Matrix<double, 3, 3>::identity()*zdm::displacement[zdm::meter];
//     [[maybe_unused]] const auto m2 = m*m;
//     return true;
// }
//
// bool test_matrix_with_units_can_multiply_with_vector()
// {
//     const auto m = zdm::la::Matrix<double, 3, 3>::identity()*zdm::displacement[zdm::meter];
//     const auto v = zdm::la::Vector<double, 3>{1.0, 2.0, 3.0};
//     [[maybe_unused]] const auto u = m*v;
//     return true;
// }
//
// bool test_matrix_with_units_can_multiply_with_vector_with_units()
// {
//     const auto m = zdm::la::Matrix<double, 3, 3>::identity()*zdm::displacement[zdm::meter];
//     const auto v = zdm::la::Vector<double, 3>{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
//     [[maybe_unused]] const auto u = m*v;
//     return true;
// }
//
// bool test_matrix_with_units_has_quadratic_form_with_vector()
// {
//     const auto m = zdm::la::Matrix<double, 3, 3>::identity()*zdm::displacement[zdm::meter];
//     const auto v = zdm::la::Vector{1.0, 2.0, 3.0};
//     [[maybe_unused]] const auto q = zdm::la::quadratic_form(m, v);
//     return true;
// }
//
// bool test_matrix_with_units_has_quadratic_form_with_vector_with_units()
// {
//     const auto m = zdm::la::Matrix<double, 3, 3>::identity()*zdm::displacement[zdm::meter];
//     const auto v = zdm::la::Vector{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
//     [[maybe_unused]] const auto q = zdm::la::quadratic_form(m, v);
//     return true;
// }

bool test_translation_can_be_constructed_from_vector_with_units()
{
    const auto v = zdm::la::Vector<double, 3>{1.0, 2.0, 3.0}*zdm::displacement[zdm::meter];
    [[maybe_unused]] const auto t = zdm::la::Translation{v};
    return true;
}

int main() {}
