#ifndef ALIGN_Z_H
#define ALIGN_Z_H
#pragma once

#include "linalg.hpp"

constexpr Matrix<double, 3, 3> rotation_matrix_align_z(
    const Vector<double, 3>& unit_vec)
{
    Matrix<double, 3, 3> matrix;
    const double u_xx = unit_vec[0]*unit_vec[0];
    const double u_yy = unit_vec[1]*unit_vec[1];
    const double u_xy = unit_vec[0]*unit_vec[1];
    const double r2 = u_xx + u_yy;
    if (r2 > 0.0)
    {
        const double scale = 1.0/r2;
        const double u_xx_norm = u_xx*scale;
        const double u_yy_norm = u_yy*scale;
        matrix[0][0] = u_yy_norm + unit_vec[2]*u_xx_norm;
        matrix[0][1] = -(1.0 - unit_vec[2])*(u_xy*scale);
        matrix[1][0] = matrix[0][1];
        matrix[1][1] = u_xx_norm + unit_vec[2]*u_yy_norm;
    }
    else
    {
        matrix[0][0] = unit_vec[2];
        matrix[0][1] = 0.0;
        matrix[1][0] = 0.0;
        matrix[1][1] = 1.0;
    }
    matrix[0][2] = -unit_vec[0];
    matrix[1][2] = -unit_vec[1];
    matrix[2][0] = unit_vec[0];
    matrix[2][1] = unit_vec[1];
    matrix[2][2] = unit_vec[2];

    return matrix;
}

constexpr Matrix<double, 3, 3> rotation_matrix_align_z_transp(
    const Vector<double, 3>& unit_vec)
{
    Matrix<double, 3, 3> matrix;
    const double u_xx = unit_vec[0]*unit_vec[0];
    const double u_yy = unit_vec[1]*unit_vec[1];
    const double u_xy = unit_vec[0]*unit_vec[1];
    const double r2 = u_xx + u_yy;
    if (r2 > 0.0)
    {
        const double scale = 1.0/r2;
        const double u_xx_norm = u_xx*scale;
        const double u_yy_norm = u_yy*scale;
        matrix[0][0] = u_yy_norm + unit_vec[2]*u_xx_norm;
        matrix[0][1] = -(1.0 - unit_vec[2])*(u_xy*scale);
        matrix[1][0] = matrix[0][1];
        matrix[1][1] = u_xx_norm + unit_vec[2]*u_yy_norm;
    }
    else
    {
        matrix[0][0] = unit_vec[2];
        matrix[0][1] = 0.0;
        matrix[1][0] = 0.0;
        matrix[1][1] = 1.0;
    }
    matrix[0][2] = unit_vec[0];
    matrix[1][2] = unit_vec[1];
    matrix[2][0] = -unit_vec[0];
    matrix[2][1] = -unit_vec[1];
    matrix[2][2] = unit_vec[2];

    return matrix;
}

#endif