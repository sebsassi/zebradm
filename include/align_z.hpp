/*
Copyright (c) 2024 Sebastian Sassi

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
#ifndef ALIGN_Z_H
#define ALIGN_Z_H
#pragma once

#include "linalg.hpp"

namespace zdm
{
namespace detail
{

[[nodiscard]] constexpr RotationMatrix<double, 3> rotation_matrix_align_z(
    const std::array<double, 3>& unit_vec) noexcept
{
    RotationMatrix<double, 3> matrix;
    const double u_xx = unit_vec[0]*unit_vec[0];
    const double u_yy = unit_vec[1]*unit_vec[1];
    const double u_xy = unit_vec[0]*unit_vec[1];
    const double r2 = u_xx + u_yy;
    if (r2 > 0.0)
    {
        const double scale = 1.0/r2;
        const double u_xx_norm = u_xx*scale;
        const double u_yy_norm = u_yy*scale;
        matrix[0, 0] = u_yy_norm + unit_vec[2]*u_xx_norm;
        matrix[0, 1] = -(1.0 - unit_vec[2])*(u_xy*scale);
        matrix[1, 0] = matrix[0, 1];
        matrix[1, 1] = u_xx_norm + unit_vec[2]*u_yy_norm;
    }
    else
    {
        matrix[0, 0] = unit_vec[2];
        matrix[0, 1] = 0.0;
        matrix[1, 0] = 0.0;
        matrix[1, 1] = 1.0;
    }
    matrix[0, 2] = -unit_vec[0];
    matrix[1, 2] = -unit_vec[1];
    matrix[2, 0] = unit_vec[0];
    matrix[2, 1] = unit_vec[1];
    matrix[2, 2] = unit_vec[2];

    return matrix;
}

[[nodiscard]] constexpr Matrix<double, 3, 3> rotation_matrix_align_z_transp(
    const std::array<double, 3>& unit_vec) noexcept
{
    RotationMatrix<double, 3, 3> matrix;
    const double u_xx = unit_vec[0]*unit_vec[0];
    const double u_yy = unit_vec[1]*unit_vec[1];
    const double u_xy = unit_vec[0]*unit_vec[1];
    const double r2 = u_xx + u_yy;
    if (r2 > 0.0)
    {
        const double scale = 1.0/r2;
        const double u_xx_norm = u_xx*scale;
        const double u_yy_norm = u_yy*scale;
        matrix[0, 0] = u_yy_norm + unit_vec[2]*u_xx_norm;
        matrix[0, 1] = -(1.0 - unit_vec[2])*(u_xy*scale);
        matrix[1, 0] = matrix[0, 1];
        matrix[1, 1] = u_xx_norm + unit_vec[2]*u_yy_norm;
    }
    else
    {
        matrix[0, 0] = unit_vec[2];
        matrix[0, 1] = 0.0;
        matrix[1, 0] = 0.0;
        matrix[1, 1] = 1.0;
    }
    matrix[0, 2] = unit_vec[0];
    matrix[1, 2] = unit_vec[1];
    matrix[2, 0] = -unit_vec[0];
    matrix[2, 1] = -unit_vec[1];
    matrix[2, 2] = unit_vec[2];

    return matrix;
}

} // namespace detail
} // namespace zdm

#endif
