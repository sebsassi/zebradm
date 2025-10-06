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
#pragma once

#include <array>
#include <cmath>

namespace zdm
{

enum class MatrixLayout
{
    row_major,
    column_major
};

enum class EulerOrder
{
    xzx,
    xyx,
    yxy,
    yzy,
    zyz,
    zxz
};

enum class TaitBryanOrder
{
    xzy,
    xyz,
    yxz,
    yzx,
    zyx,
    zxy
};

enum class RotationConvention
{
    intrinsic,
    extrinsic
};

enum class TransformAction
{
    active,
    passive
};

template <typename T, std::size_t N>
using Vector = std::array<T, N>;

template <
    typename T, std::size_t N, std::size_t M,
    TransformAction action_param = TransformAction::passive,
    MatrixLayout layout_param = MatrixLayout::column_major
>
struct Matrix
{
    using value_type = T;
    using index_type = std::size_t;
    using size_type = std::size_t;

    static constexpr TransformAction action = action_param;
    static constexpr MatrixLayout layout = layout_param;
    std::array<T, N*M> array;

    [[nodiscard]] constexpr T& operator[](std::size_t i, std::size_t j) noexcept
    {
        if constexpr (layout == MatrixLayout::row_major)
            return array[M*i + j];
        else
            return array[N*j + i];
    }

    [[nodiscard]] constexpr const T& operator[](std::size_t i, std::size_t j) const noexcept
    {
        if constexpr (layout == MatrixLayout::row_major)
            return array[M*i + j];
        else
            return array[N*j + i];
    }
};

template <
    typename T, std::size_t N,
    TransformAction action_param = TransformAction::passive,
    MatrixLayout layout_param = MatrixLayout::column_major
>
class RotationMatrix
{
public:
    using value_type = T;
    using index_type = std::size_t;
    using size_type = std::size_t;

    static constexpr TransformAction action = action_param;
    static constexpr MatrixLayout layout = layout_param;

    template <
        typename U,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 2, action, layout> from_angle(U angle) noexcept
    {
        const U cos_angle = std::cos(angle);
        const U sin_angle = std::sin(angle);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                cos_angle, -sin_angle,
                sin_angle, cos_angle
            });
        else
            return RotationMatrix({
                cos_angle, sin_angle,
                -sin_angle, cos_angle
            });
    }

    template <
        typename U,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout> axis_x(U angle) noexcept
    {
        const U ca = std::cos(angle);
        const U sa = std::sin(angle);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                 1.0,  0.0,  0.0,
                 0.0,  ca,  -sa,
                 0.0,  sa,   ca
            });
        else
            return RotationMatrix({
                 1.0,  0.0,  0.0,
                 0.0,  ca,   sa,
                 0.0, -sa,   ca
            });
    }

    template <
        typename U,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout> axis_y(U angle) noexcept
    {
        const U ca = std::cos(angle);
        const U sa = std::sin(angle);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                 ca,   0.0, -sa,
                 0.0,  1.0,  0.0,
                 sa,   0.0,  ca
            });
        else
            return RotationMatrix({
                 ca,   0.0,  sa,
                 0.0,  1.0,  0.0,
                -sa,   0.0,  ca
            });
    }

    template <
        typename U,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout> axis_z(U angle) noexcept
    {
        const U ca = std::cos(angle);
        const U sa = std::sin(angle);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                 ca,  -sa,   0.0,
                 sa,   ca,   0.0,
                 0.0,  0.0,  1.0
            });
        else
            return RotationMatrix({
                 ca,   sa,   0.0,
                -sa,   ca,   0.0,
                 0.0,  0.0,  1.0
            });
    }

    template <
        typename U,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout> 
    axis(std::array<U, 3> axis, U angle) noexcept
    {
        axis = normalize(axis);
        const U x = axis[0];
        const U y = axis[1];
        const U z = axis[2];
        const U xx = axis[0]*axis[0];
        const U yy = axis[1]*axis[1];
        const U zz = axis[2]*axis[2];
        const U xy = axis[0]*axis[1];
        const U xz = axis[0]*axis[2];
        const U yz = axis[1]*axis[2];
        const U cos_angle = std::cos(angle);
        const U sin_angle = std::sin(angle);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                xx*(1.0 - cos_angle) + cos_angle, xy*(1.0 - cos_angle) - z*sin_angle, xz*(1.0 - cos_angle) + y*sin_angle,
                xy*(1.0 - cos_angle) + z*sin_angle, yy*(1.0 - cos_angle) + cos_angle, yz*(1.0 - cos_angle) - x*sin_angle,
                xz*(1.0 - cos_angle) - y*sin_angle, yz*(1.0 - cos_angle) + x*sin_angle, zz*(1.0 - cos_angle) + cos_angle
            });
        else
            return RotationMatrix({
                xx*(1.0 - cos_angle) + cos_angle, xy*(1.0 - cos_angle) + z*sin_angle, xz*(1.0 - cos_angle) - y*sin_angle,
                xy*(1.0 - cos_angle) - z*sin_angle, yy*(1.0 - cos_angle) + cos_angle, yz*(1.0 - cos_angle) + x*sin_angle,
                xz*(1.0 - cos_angle) + y*sin_angle, yz*(1.0 - cos_angle) - x*sin_angle, zz*(1.0 - cos_angle) + cos_angle
            });
    }
    
    // NOTE: composite rotations are in order of matrix multiplication, e.g.
    // `composite_axes_xyz(alpha, beta, gamma)` corresponds to the rotation
    // `R_x(alpha)R_y(beta)R_z(gamma)`. This notably means that the order of
    // angles is reveresed relative to the usual Euler angles, where `alpha`
    // would denote the angle of the first rotation

    template <
        typename U, EulerOrder order,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes_xy(U alpha, U beta) noexcept
    {
        const U ca = std::cos(alpha);
        const U sa = std::sin(alpha);
        const U cb = std::cos(beta);
        const U sb = std::sin(beta);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                // TODO
            });
        else
            return RotationMatrix({
                // TODO
            });
    }

    template <
        typename U, EulerOrder order,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes_xz(U alpha, U beta) noexcept
    {
        const U ca = std::cos(alpha);
        const U sa = std::sin(alpha);
        const U cb = std::cos(beta);
        const U sb = std::sin(beta);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                // TODO
            });
        else
            return RotationMatrix({
                // TODO
            });
    }

    template <
        typename U, EulerOrder order,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes_yx(U alpha, U beta) noexcept
    {
        const U ca = std::cos(alpha);
        const U sa = std::sin(alpha);
        const U cb = std::cos(beta);
        const U sb = std::sin(beta);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                // TODO
            });
        else
            return RotationMatrix({
                // TODO
            });
    }

    template <
        typename U, EulerOrder order,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes_yz(U alpha, U beta) noexcept
    {
        const U ca = std::cos(alpha);
        const U sa = std::sin(alpha);
        const U cb = std::cos(beta);
        const U sb = std::sin(beta);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                // TODO
            });
        else
            return RotationMatrix({
                // TODO
            });
    }

    template <
        typename U, EulerOrder order,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes_zx(U alpha, U beta) noexcept
    {
        const U ca = std::cos(alpha);
        const U sa = std::sin(alpha);
        const U cb = std::cos(beta);
        const U sb = std::sin(beta);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                // TODO
            });
        else
            return RotationMatrix({
                // TODO
            });
    }

    template <
        typename U, EulerOrder order,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes_zy(U alpha, U beta) noexcept
    {
        const U ca = std::cos(alpha);
        const U sa = std::sin(alpha);
        const U cb = std::cos(beta);
        const U sb = std::sin(beta);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                // TODO
            });
        else
            return RotationMatrix({
                // TODO
            });
    }

    template <
        typename U, EulerOrder order,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes_xyx(U alpha, U beta, U gamma) noexcept
    {
        const U ca = std::cos(alpha);
        const U sa = std::sin(alpha);
        const U cb = std::cos(beta);
        const U sb = std::sin(beta);
        const U cg = std::cos(gamma);
        const U sg = std::sin(gamma);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                 cb,                sb*sg,             cg*sb,
                 sa*sb,             ca*cg - cb*sa*sg, -ca*sg - cb*cg*sa,
                -ca*sb,             cg*sa + ca*cb*sg,  ca*cb*cg - sa*sg
            });
        else
            return RotationMatrix({
                 cb,                sa*sb,            -ca*sb,
                 sb*sg,             ca*cg - cb*sa*sg,  cg*sa + ca*cb*sg,
                 cg*sb,            -ca*sg - cb*cg*sa,  ca*cb*cg - sa*sg
            });
    }

    template <
        typename U, EulerOrder order,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes_xzx(U alpha, U beta, U gamma) noexcept
    {
        const U ca = std::cos(alpha);
        const U sa = std::sin(alpha);
        const U cb = std::cos(beta);
        const U sb = std::sin(beta);
        const U cg = std::cos(gamma);
        const U sg = std::sin(gamma);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                 cb,               -cg*sb,             sb*sg,
                 ca*sb,             ca*cb*cg - sa*sg, -cg*sa - ca*cb*sg,
                 sa*sb,             ca*sg + cb*cg*sa,  ca*cg - cb*sa*sg
            });
        else
            return RotationMatrix({
                 cb,                ca*sb,             sa*sb,
                -cg*sb,             ca*cb*cg - sa*sg,  ca*sg + cb*cg*sa,
                 sb*sg,            -cg*sa - ca*cb*sg,  ca*cg - cb*sa*sg
            });
    }

    template <
        typename U, EulerOrder order,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes_yxy(U alpha, U beta, U gamma) noexcept
    {
        const U ca = std::cos(alpha);
        const U sa = std::sin(alpha);
        const U cb = std::cos(beta);
        const U sb = std::sin(beta);
        const U cg = std::cos(gamma);
        const U sg = std::sin(gamma);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                 ca*cg - cb*sa*sg,  sa*sb,             ca*sg + cb*cg*sa,
                 sb*sg,             cb,               -cg*sb,
                -cg*sa - ca*cg*sg,  ca*sb,             ca*cb*cg - ca*sg
            });
        else
            return RotationMatrix({
                 ca*cg - cb*sa*sg,  sb*sg,            -cg*sa - ca*cg*sg,
                 sa*sb,             cb,                ca*sb,
                 ca*sg + cb*cg*sa, -cg*sb,             ca*cb*cg - ca*sg
            });
    }

    template <
        typename U, EulerOrder order,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes_yzy(U alpha, U beta, U gamma) noexcept
    {
        const U ca = std::cos(alpha);
        const U sa = std::sin(alpha);
        const U cb = std::cos(beta);
        const U sb = std::sin(beta);
        const U cg = std::cos(gamma);
        const U sg = std::sin(gamma);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                 ca*cb*cg - sa*sg, -ca*sb,             cg*sa + ca*cb*sg,
                 cg*sb,             cb,                sb*sg,
                -ca*sg - cb*cg*sa,  sa*sb,             ca*cg - cb*sa*sg
            });
        else
            return RotationMatrix({
                 ca*cb*cg - sa*sg,  cg*sb,            -ca*sg - cb*cg*sa,
                -ca*sb,             cb,                sa*sb,
                 cg*sa + ca*cb*sg,  sb*sg,             ca*cg - cb*sa*sg
            });
    }

    template <
        typename U, EulerOrder order,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes_zxz(U alpha, U beta, U gamma) noexcept
    {
        const U ca = std::cos(alpha);
        const U sa = std::sin(alpha);
        const U cb = std::cos(beta);
        const U sb = std::sin(beta);
        const U cg = std::cos(gamma);
        const U sg = std::sin(gamma);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                 ca*cg - cb*sa*sg, -ca*sg - cb*cg*sa,  sa*sb,
                 cg*sa + ca*cb*sg,  ca*cb*cg - sa*sg, -ca*sb,
                 sb*sg,             cg*sb,             cb
            });
        else
            return RotationMatrix({
                 ca*cg - cb*sa*sg,  cg*sa + ca*cb*sg,  sb*sg,
                -ca*sg - cb*cg*sa,  ca*cb*cg - sa*sg,  cg*sb,
                 sa*sb,            -ca*sb,             cb
            });
    }

    template <
        typename U, EulerOrder order,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes_zyz(U alpha, U beta, U gamma) noexcept
    {
        const U ca = std::cos(alpha);
        const U sa = std::sin(alpha);
        const U cb = std::cos(beta);
        const U sb = std::sin(beta);
        const U cg = std::cos(gamma);
        const U sg = std::sin(gamma);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                 ca*cb*cg - sa*sg, -cg*sa - ca*cb*sg,  ca*sb,
                 ca*sg + cb*cg*sa,  ca*cg - cb*sa*sg,  sa*sb,
                -cg*sb,             sb*sg,             cb
            });
        else
            return RotationMatrix({
                 ca*cb*cg - sa*sg,  ca*sg + cb*cg*sa, -cg*sb,
                -cg*sa - ca*cb*sg,  ca*cg - cb*sa*sg,  sb*sg,
                 ca*sb,             sa*sb,             cb
            });
    }

    template <
        typename U, EulerOrder order,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes_xyz(U alpha, U beta, U gamma) noexcept
    {
        const U ca = std::cos(alpha);
        const U sa = std::sin(alpha);
        const U cb = std::cos(beta);
        const U sb = std::sin(beta);
        const U cg = std::cos(gamma);
        const U sg = std::sin(gamma);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                 cb*cg,            -cb*sg,             sb,
                 ca*sg + cg*sa*sb,  ca*cg - sa*sb*sg, -cb*sa,
                 sa*sg - ca*cg*sb,  cg*sa + ca*sb*sg,  ca*cb
            });
        else
            return RotationMatrix({
                 cb*cg,             ca*sg + cg*sa*sb,  sa*sg - ca*cg*sb,
                -cb*sg,             ca*cg - sa*sb*sg,  cg*sa + ca*sb*sg,
                 sb,               -cb*sa,             ca*cb
            });
    }

    template <
        typename U, EulerOrder order,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes_xzy(U alpha, U beta, U gamma) noexcept
    {
        const U ca = std::cos(alpha);
        const U sa = std::sin(alpha);
        const U cb = std::cos(beta);
        const U sb = std::sin(beta);
        const U cg = std::cos(gamma);
        const U sg = std::sin(gamma);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                 cb*cg,            -sb,                cb*sg,
                 sa*sg + ca*cg*sb,  ca*cb,             ca*sb*sg - cg*sa,
                 cg*sa*sb - ca*sg,  cb*sa,             ca*cg + sa*sb*sg
            });
        else
            return RotationMatrix({
                 cb*cg,             sa*sg + ca*cg*sb,  cg*sa*sb - ca*sg,
                -sb,                ca*cb,             cb*sa,
                 cb*sg,             ca*sb*sg - cg*sa,  ca*cg + sa*sb*sg
            });
    }

    template <
        typename U, EulerOrder order,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes_yxz(U alpha, U beta, U gamma) noexcept
    {
        const U ca = std::cos(alpha);
        const U sa = std::sin(alpha);
        const U cb = std::cos(beta);
        const U sb = std::sin(beta);
        const U cg = std::cos(gamma);
        const U sg = std::sin(gamma);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                 ca*cg + sa*sb*sg,  cg*sa*sb - ca*sg,  cb*sa,
                 cb*sg,             cb*cg,            -sb,
                 ca*sb*sg - cg*sa,  ca*cg*sb + sa*sg,  ca*cb
            });
        else
            return RotationMatrix({
                 ca*cg + sa*sb*sg,  cb*sg,             ca*sb*sg - cg*sa,
                 cg*sa*sb - ca*sg,  cb*cg,             ca*cg*sb + sa*sg,
                 cb*sa,            -sb,                ca*cb
            });
    }

    template <
        typename U, EulerOrder order,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes_yzx(U alpha, U beta, U gamma) noexcept
    {
        const U ca = std::cos(alpha);
        const U sa = std::sin(alpha);
        const U cb = std::cos(beta);
        const U sb = std::sin(beta);
        const U cg = std::cos(gamma);
        const U sg = std::sin(gamma);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                 ca*cb,             sa*sg - ca*cg*sb,  cg*sa + ca*sb*sg,
                 sb,                cb*cg,            -cb*sg,
                -cb*sa,             ca*sg + cg*sa*sb,  ca*cg - sa*sb*sg
            });
        else
            return RotationMatrix({
                 ca*cb,             sb,               -cb*sa,
                 sa*sg - ca*cg*sb,  cb*cg,             ca*sg + cg*sa*sb,
                 cg*sa + ca*sb*sg, -cb*sg,             ca*cg - sa*sb*sg
            });
    }

    template <
        typename U, EulerOrder order,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes_zxy(U alpha, U beta, U gamma) noexcept
    {
        const U ca = std::cos(alpha);
        const U sa = std::sin(alpha);
        const U cb = std::cos(beta);
        const U sb = std::sin(beta);
        const U cg = std::cos(gamma);
        const U sg = std::sin(gamma);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                 ca*cg - sa*sb*sg, -cb*sa,             ca*sg + cg*sa*sb,
                 cg*sa + ca*sb*sg,  ca*cb,             sa*sg - ca*cg*sb,
                -cb*sg,             sb,                cb*cg,
            });
        else
            return RotationMatrix({
                 ca*cg - sa*sb*sg,  cg*sa + ca*sb*sg, -cb*sg,
                -cb*sa,             ca*cb,             sb,
                 ca*sg + cg*sa*sb,  sa*sg - ca*cg*sb,  cb*cg,
            });
    }

    template <
        typename U, EulerOrder order,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes_zyx(U alpha, U beta, U gamma) noexcept
    {
        const U ca = std::cos(alpha);
        const U sa = std::sin(alpha);
        const U cb = std::cos(beta);
        const U sb = std::sin(beta);
        const U cg = std::cos(gamma);
        const U sg = std::sin(gamma);
        if constexpr (
                (layout == MatrixLayout::row_major && action == TransformAction::active)
                || (layout == MatrixLayout::column_major && action == TransformAction::passive))
            return RotationMatrix({
                 ca*cb,             ca*sb*sg - cg*sa,  sa*sg + ca*cg*sb,
                 cb*sa,             ca*cg + sa*sb*sg,  cg*sa*sb - ca*sg,
                -sb,                cb*sg,             cb*cg
            });
        else
            return RotationMatrix({
                 ca*cb,             cb*sa,            -sb,
                 ca*sb*sg - cg*sa,  ca*cg + sa*sb*sg,  cb*sg,
                 sa*sg + ca*cg*sb,  cg*sa*sb - ca*sg,  cb*cg
            });
    }

    [[nodiscard]] constexpr T&
    operator[](std::size_t i, std::size_t j) noexcept { return matrix[i, j]; }

    [[nodiscard]] constexpr const T&
    operator[](std::size_t i, std::size_t j) const noexcept { return matrix[i, j]; }

private:
    RotationMatrix(std::array<T, N*N> array): matrix{array} {}

    Matrix<T, N, N, action, layout> matrix;
};

template <typename T, std::size_t N, MatrixLayout layout_param = MatrixLayout::column_major>
class AffineMatrix
{
public:
    using value_type = T;
    using index_type = std::size_t;
    using size_type = std::size_t;

    static constexpr MatrixLayout layout = layout_param;

    [[nodiscard]] constexpr T& operator[](std::size_t i, std::size_t j) noexcept { return matrix[i, j]; }
    [[nodiscard]] constexpr const T& operator[](std::size_t i, std::size_t j) const noexcept { return matrix[i, j]; }

private:
    Matrix<T, N, N, layout> matrix;
};

namespace detail
{

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> add_(
    const std::array<T, sizeof...(idx)>& a, const std::array<T, sizeof...(idx)>& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] + b[idx])...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> add_(
    const T& a, const std::array<T, sizeof...(idx)>& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a + b[idx])...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> add_(
    const std::array<T, sizeof...(idx)>& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] + b)...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> sub_(
    const std::array<T, sizeof...(idx)>& a, const std::array<T, sizeof...(idx)>& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] - b[idx])...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> sub_(
    const T& a, const std::array<T, sizeof...(idx)>& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a - b[idx])...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> sub_(
    const std::array<T, sizeof...(idx)>& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] - b)...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> mul_(
    const std::array<T, sizeof...(idx)>& a, const std::array<T, sizeof...(idx)>& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]*b[idx])...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> mul_(
    const T& a, const std::array<T, sizeof...(idx)>& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a*b[idx])...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> mul_(
    const std::array<T, sizeof...(idx)>& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]*b)...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> div_(
    const std::array<T, sizeof...(idx)>& a, const std::array<T, sizeof...(idx)>& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]/b[idx])...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> div_(
    const T& a, const std::array<T, sizeof...(idx)>& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a/b[idx])...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr std::array<T, sizeof...(idx)> div_(
    const std::array<T, sizeof...(idx)>& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]/b)...}};
}

template <typename T, std::size_t... idx>
[[nodiscard]] constexpr T dot_(
    const std::array<T, sizeof...(idx)>& a, const std::array<T, sizeof...(idx)>& b, 
    std::index_sequence<idx...>) noexcept
{
    return ((a[idx]*b[idx]) + ...);
}

} // namespace detail

template <typename T, std::size_t N>
[[nodiscard]] constexpr T dot(const std::array<T, N>& a, const std::array<T, N>& b) noexcept
{
    return detail::dot_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] inline T length(const std::array<T, N>& a) noexcept
{
    return std::sqrt(dot(a, a));
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
add(const std::array<T, N>& a, const std::array<T, N>& b) noexcept
{
    return detail::add_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
sub(const std::array<T, N>& a, const std::array<T, N>& b) noexcept
{
    return detail::sub_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
mul(const std::array<T, N>& a, const std::array<T, N>& b) noexcept
{
    return detail::mul_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
div(const std::array<T, N>& a, const std::array<T, N>& b) noexcept
{
    return detail::div_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
constexpr std::array<T, N>&
add_assign(std::array<T, N>& a, const std::array<T, N>& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] += b[i];
    return a;
}

template <typename T, std::size_t N>
constexpr std::array<T, N>&
sub_assign(std::array<T, N>& a, const std::array<T, N>& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] -= b[i];
    return a;
}

template <typename T, std::size_t N>
constexpr std::array<T, N>&
mul_assign(std::array<T, N>& a, const std::array<T, N>& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] *= b[i];
    return a;
}

template <typename T, std::size_t N>
constexpr std::array<T, N>&
div_assign(std::array<T, N>& a, const std::array<T, N>& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] /= b[i];
    return a;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
add(const T& a, const std::array<T, N>& b) noexcept
{
    return detail::add_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
sub(const T& a, const std::array<T, N>& b) noexcept
{
    return detail::sub_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
mul(const T& a, const std::array<T, N>& b) noexcept
{
    return detail::mul_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
div(const T& a, const std::array<T, N>& b) noexcept
{
    return detail::div_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
add(const std::array<T, N>& a, const T& b) noexcept
{
    return detail::add_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
sub(const std::array<T, N>& a, const T& b) noexcept
{
    return detail::sub_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
mul(const std::array<T, N>& a, const T& b) noexcept
{
    return detail::mul_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N>
div(const std::array<T, N>& a, const T& b) noexcept
{
    return detail::div_(a, b, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
constexpr std::array<T, N>& add_assign(std::array<T, N>& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] += b;
    return a;
}

template <typename T, std::size_t N>
constexpr std::array<T, N>& sub_assign(std::array<T, N>& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] -= b;
    return a;
}

template <typename T, std::size_t N>
constexpr std::array<T, N>& mul_assign(std::array<T, N>& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] *= b;
    return a;
}

template <typename T, std::size_t N>
constexpr std::array<T, N>& div_assign(std::array<T, N>& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        a[i] /= b;
    return a;
}

template <typename T, std::size_t N, std::size_t M, MatrixLayout layout>
[[nodiscard]] constexpr std::array<T, N>
matmul(const Matrix<T, N, M, layout>& mat, const std::array<T, M>& vec) noexcept
{
    std::array<T, N> res{};
    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = 0; j < M; ++j)
            res[i] += mat[i, j]*vec[j];
    }

    return res;
}

template <typename T, std::size_t N, std::size_t M, std::size_t L, MatrixLayout layout>
[[nodiscard]] constexpr Matrix<T, N, L, layout>
matmul(const Matrix<T, N, M, layout>& a, const Matrix<T, M, L, layout>& b) noexcept
{
    Matrix<T, N, L, layout> res{};
    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = 0; j < L; ++j)
        {
            for (std::size_t k = 0; k < M; ++k)
                res[i, j] += a[i, k]*b[k, j];
        }
    }

    return res;
}

template <typename T, std::size_t N, std::size_t M, MatrixLayout layout>
[[nodiscard]] constexpr Matrix<T, M, N, layout>
transpose(const Matrix<T, N, M, layout>& matrix)
{
    Matrix<T, M, N, layout> res{};
    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = 0; j < M; ++j)
            res[j, i] = matrix[i, j];
    }
    return res;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::array<T, N> normalize(const std::array<T, N>& a) noexcept
{
    const T norm = length(a);
    if (norm == T{}) return std::array<T, N>{};

    return mul((1.0/norm), a);
}

} // namespace zdm
