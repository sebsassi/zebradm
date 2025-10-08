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

#include <concepts>
#include <type_traits>
#include <array>
#include <cmath>

namespace zdm
{

enum class Axis { x, y, z };

enum class MatrixLayout
{
    row_major,
    column_major
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

template <typename T>
concept matrix_like = std::same_as<decltype(T::shape), std::array<typename T::size_type, 2>>
    && requires (T matrix, typename T::index_type i, T::index_type j)
    {
        { matrix[i, j] } -> std::same_as<std::add_lvalue_reference_t<typename T::value_type>>;
    };

template <typename T>
concept vector_like = std::same_as<decltype(std::tuple_size_v<T>), typename T::size_type>
    && requires (T vector, typename T::index_type i)
    {
        { vector[i] } -> std::same_as<std::add_lvalue_reference_t<typename T::value_type>>;
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
    static constexpr std::array<size_type, 2> shape = {N, M};

    std::array<T, N*M> array;

    template <
        typename U, std::size_t K,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr Matrix<U, K, K, action, layout> identity()
    {
        Matrix<U, K, K, action, layout> res{};
        for (std::size_t i = 0; i < K; ++i)
            res[i, i] = 1.0;
        return res;
    }

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
    static constexpr std::array<size_type, 2> shape = {N, N};

    constexpr explicit RotationMatrix() = default;
    constexpr explicit RotationMatrix(std::array<T, N*N> array): m_matrix{array} {}

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
        typename U, Axis axis,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout> coordinate_axis(U angle) noexcept
    {
        if constexpr (axis == Axis::x)
            return axis_x(angle);
        if constexpr (axis == Axis::y)
            return axis_y(angle);
        if constexpr (axis == Axis::z)
            return axis_z(angle);
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
        typename U, Axis axis_alpha, Axis axis_beta,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes(U alpha, U beta) noexcept
    {
        if constexpr (axis_alpha == Axis::x && axis_beta == Axis::y)
            return composite_axes_xy(alpha, beta);
        if constexpr (axis_alpha == Axis::x && axis_beta == Axis::z)
            return composite_axes_xz(alpha, beta);
        if constexpr (axis_alpha == Axis::y && axis_beta == Axis::x)
            return composite_axes_yx(alpha, beta);
        if constexpr (axis_alpha == Axis::y && axis_beta == Axis::z)
            return composite_axes_yz(alpha, beta);
        if constexpr (axis_alpha == Axis::z && axis_beta == Axis::x)
            return composite_axes_zx(alpha, beta);
        if constexpr (axis_alpha == Axis::z && axis_beta == Axis::y)
            return composite_axes_zy(alpha, beta);
    }

    template <
        typename U, Axis axis_alpha, Axis axis_beta, Axis axis_gamma,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RotationMatrix<U, 3, action, layout>
    composite_axes(U alpha, U beta, U gamma) noexcept
    {
        if constexpr (axis_alpha == Axis::x && axis_beta == Axis::y && axis_gamma == Axis::x)
            return composite_axes_xyx(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::x && axis_beta == Axis::z && axis_gamma == Axis::x)
            return composite_axes_xzx(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::y && axis_beta == Axis::x && axis_gamma == Axis::y)
            return composite_axes_yxy(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::y && axis_beta == Axis::z && axis_gamma == Axis::y)
            return composite_axes_yzy(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::z && axis_beta == Axis::x && axis_gamma == Axis::z)
            return composite_axes_zxz(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::z && axis_beta == Axis::y && axis_gamma == Axis::z)
            return composite_axes_zyz(alpha, beta, gamma);

        if constexpr (axis_alpha == Axis::x && axis_beta == Axis::y && axis_gamma == Axis::z)
            return composite_axes_xyz(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::x && axis_beta == Axis::z && axis_gamma == Axis::y)
            return composite_axes_xzy(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::y && axis_beta == Axis::x && axis_gamma == Axis::z)
            return composite_axes_yxz(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::y && axis_beta == Axis::z && axis_gamma == Axis::x)
            return composite_axes_yzx(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::z && axis_beta == Axis::x && axis_gamma == Axis::y)
            return composite_axes_zxy(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::z && axis_beta == Axis::y && axis_gamma == Axis::x)
            return composite_axes_zyx(alpha, beta, gamma);
    }

    [[nodiscard]] constexpr operator Matrix<T, N, N, action, layout>() noexcept { return m_matrix; }
    [[nodiscard]] constexpr operator std::array<T, N*N>() noexcept { return m_matrix.array; }

    [[nodiscard]] constexpr T&
    operator[](std::size_t i, std::size_t j) noexcept { return m_matrix[i, j]; }

    [[nodiscard]] constexpr const T&
    operator[](std::size_t i, std::size_t j) const noexcept { return m_matrix[i, j]; }

    [[nodiscard]] constexpr RotationMatrix inverse() const noexcept
    {
        return RotationMatrix(transpose(*m_matrix));
    }

private:
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
                 cb,     0.0,    sb,
                 sa*sb,  ca,    -cb*sa,
                -ca*sb,  sa,     ca*cb
            });
        else
            return RotationMatrix({
                 cb,     sa*sb,  -ca*sb,
                 0.0,    ca,     sa,
                 sb,    -cb*sa,  ca*cb
            });
    }

    template <
        typename U,
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
                 cb,    -sb,     0.0,
                 ca*sb,  ca*cb, -sa,
                 sa*sb,  cb*sa,  ca
            });
        else
            return RotationMatrix({
                 cb,     ca*sb,  sa*sb,
                -sb,     ca*cb,  cb*sa,
                 0.0,   -sa,     ca
            });
    }

    template <
        typename U,
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
                 ca,     sa*sb,  cb*sa,
                 0.0,    cb,    -sb,
                -sa,     ca*sb,  ca*cb
            });
        else
            return RotationMatrix({
                 ca,     0.0,   -sa,
                 sa*sb,  cb,     ca*sb,
                 cb*sa, -sb,     ca*cb
            });
    }

    template <
        typename U,
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
                 ca*cb, -ca*sb,  sa,
                 sb,     cb,     0.0,
                -cb*sa,  sa*sb,  ca
            });
        else
            return RotationMatrix({
                 ca*cb,  sb,    -cb*sa,
                -ca*sb,  cb,     sa*sb,
                 sa,     0.0,    ca
            });
    }

    template <
        typename U,
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
                 ca,    -cb*sa,  sa*sb,
                 sa,     ca*cb, -ca*sb,
                 0.0,    sb,     cb
            });
        else
            return RotationMatrix({
                 ca,     sa,     0.0,
                -cb*sa,  ca*cb,  sb,
                 sa*sb, -ca*sb,  cb
            });
    }

    template <
        typename U,
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
                 ca*cb, -sa,     ca*sb,
                 cb*sa,  ca,     sa*sb,
                -sb,     0.0,    cb
            });
        else
            return RotationMatrix({
                 ca*cb,  cb*sa, -sb,
                -sa,     ca,     0.0,
                 ca*sb,  sa*sb,  cb
            });
    }

    template <
        typename U,
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
        typename U,
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
        typename U,
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
        typename U,
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
        typename U,
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
        typename U,
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
        typename U,
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
        typename U,
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
        typename U,
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
        typename U,
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
        typename U,
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
        typename U,
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

    Matrix<T, N, N, action, layout> m_matrix = Matrix<T, N, N, action, layout>::identity();
};

template <
    typename T, std::size_t N,
    TransformAction action_param = TransformAction::passive,
    MatrixLayout layout_param = MatrixLayout::column_major>
class RigidMatrix
{
public:
    using value_type = T;
    using index_type = std::size_t;
    using size_type = std::size_t;

    static constexpr TransformAction action = action_param;
    static constexpr MatrixLayout layout = layout_param;
    static constexpr std::array<size_type, 2> shape = {N + 1, N + 1};

    constexpr explicit RigidMatrix() = default;
    constexpr explicit RigidMatrix(std::array<T, (N + 1)*(N + 1)> array): m_matrix{array} {}

    template <
        typename U, std::size_t M,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RigidMatrix<U, M, action, layout> from(
        RotationMatrix<U, M, action, layout> rotation, std::array<U, M> translation) noexcept
    {
        RigidMatrix<U, M, action, layout> res;
        for (std::size_t i = 0; i < M; ++i)
        {
            for (std::size_t j = 0; j < M; ++j)
                res[i, j] = rotation[i, j];
            res[i, M] = translation[i];
        }
        res[M, M] = 1.0;
        return res;
    }

    template <
        typename U, std::size_t M,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RigidMatrix<U, M, action, layout> from(
        RotationMatrix<U, M, action, layout> rotation) noexcept
    {
        RigidMatrix<U, M, action, layout> res;
        for (std::size_t i = 0; i < M; ++i)
        {
            for (std::size_t j = 0; j < M; ++j)
                res[i, j] = rotation[i, j];
        }
        res[M, M] = 1.0;
        return res;
    }

    template <
        typename U, std::size_t M,
        TransformAction action = TransformAction::passive,
        MatrixLayout layout = MatrixLayout::column_major
    >
    [[nodiscard]] static constexpr RigidMatrix<U, M, action, layout> from(
        std::array<U, M> translation) noexcept
    {
        RigidMatrix<U, M, action, layout> res;
        for (std::size_t i = 0; i < M; ++i)
            res[i, M] = translation[i];
        res[M, M] = 1.0;
        return res;
    }

    [[nodiscard]] constexpr T&
    operator[](std::size_t i, std::size_t j) noexcept { return m_matrix[i, j]; }

    [[nodiscard]] constexpr const T&
    operator[](std::size_t i, std::size_t j) const noexcept { return m_matrix[i, j]; }

    [[nodiscard]] constexpr RigidMatrix inverse() const noexcept
    {
        const auto res = RigidMatrix::from(extract_rotation().inverse());
        return RigidMatrix::from(inverse_rotation, matmul(inverse_rotation, extract_translation()));
    }

private:
    Matrix<T, N + 1, N + 1, action, layout> m_matrix = Matrix<T, N, N, action, layout>::identity();
};

namespace detail
{

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T add_(
    const T& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] + b[idx])...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T add_(
    const typename T::value_type& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a + b[idx])...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T add_(
    const T& a, const typename T::value_type& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] + b)...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T sub_(
    const T& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] - b[idx])...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T sub_(
    const typename T::value_type& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a - b[idx])...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T sub_(
    const T& a, const typename T::value_type& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx] - b)...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T mul_(
    const T& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]*b[idx])...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T mul_(
    const typename T::value_type& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a*b[idx])...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T mul_(
    const T& a, const typename T::value_type& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]*b)...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T div_(
    const T& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]/b[idx])...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T div_(
    const typename T::value_type& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a/b[idx])...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T div_(
    const T& a, const typename T::value_type& b, 
    std::index_sequence<idx...>) noexcept
{
    return {{(a[idx]/b)...}};
}

template <vector_like T, std::size_t... idx>
[[nodiscard]] constexpr T dot_(
    const T& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return ((a[idx]*b[idx]) + ...);
}

} // namespace detail

template <vector_like T>
[[nodiscard]] constexpr T dot(const T& a, const T& b) noexcept
{
    return detail::dot_(a, b, std::make_index_sequence<std::_tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] inline T length(const T& a) noexcept
{
    return std::sqrt(dot(a, a));
}

template <vector_like T>
[[nodiscard]] constexpr T
add(const T& a, const T& b) noexcept
{
    return detail::add_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
sub(const T& a, const T& b) noexcept
{
    return detail::sub_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
mul(const T& a, const T& b) noexcept
{
    return detail::mul_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
div(const T& a, const T& b) noexcept
{
    return detail::div_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
constexpr T&
add_assign(T& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] += b[i];
    return a;
}

template <vector_like T>
constexpr T&
sub_assign(T& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] -= b[i];
    return a;
}

template <vector_like T>
constexpr T&
mul_assign(T& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] *= b[i];
    return a;
}

template <vector_like T>
constexpr T&
div_assign(T& a, const T& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] /= b[i];
    return a;
}

template <vector_like T>
[[nodiscard]] constexpr T
add(const typename T::value_type& a, const T& b) noexcept
{
    return detail::add_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
sub(const typename T::value_type& a, const T& b) noexcept
{
    return detail::sub_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
mul(const typename T::value_type& a, const T& b) noexcept
{
    return detail::mul_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
div(const typename T::value_type& a, const T& b) noexcept
{
    return detail::div_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
add(const T& a, const typename T::value_type& b) noexcept
{
    return detail::add_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
sub(const T& a, const typename T::value_type& b) noexcept
{
    return detail::sub_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
mul(const T& a, const typename T::value_type& b) noexcept
{
    return detail::mul_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
[[nodiscard]] constexpr T
div(const T& a, const typename T::value_type& b) noexcept
{
    return detail::div_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
constexpr T& add_assign(T& a, const typename T::value_type& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] += b;
    return a;
}

template <vector_like T>
constexpr T& sub_assign(T& a, const typename T::value_type& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] -= b;
    return a;
}

template <vector_like T>
constexpr T& mul_assign(T& a, const typename T::value_type& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] *= b;
    return a;
}

template <vector_like T>
constexpr T& div_assign(T& a, const typename T::value_type& b) noexcept
{
    for (std::size_t i = 0; i < std::tuple_size_v<T>; ++i)
        a[i] /= b;
    return a;
}

template <matrix_like T, vector_like U>
[[nodiscard]] constexpr U
matmul(const T& mat, const U& vec) noexcept
{
    U res{};
    for (std::size_t i = 0; i < T::shape[0]; ++i)
    {
        for (std::size_t j = 0; j < T::shape[1]; ++j)
            res[i] += mat[i, j]*vec[j];
    }

    return res;
}

template <matrix_like T>
    requires (T::shape[0] == T::shape[1])
[[nodiscard]] constexpr T
matmul(const T& a, const T& b) noexcept
{
    constexpr auto dimension = T::shape[0];
    T res{};
    for (std::size_t i = 0; i < dimension; ++i)
    {
        for (std::size_t j = 0; j < dimension; ++j)
        {
            for (std::size_t k = 0; k < dimension; ++k)
                res[i, j] += a[i, k]*b[k, j];
        }
    }
}

template <matrix_like T, matrix_like U>
[[nodiscard]] constexpr Matrix<T, T::shape[0], U::shape[1], T::layout>
matmul(const T& a, const U& b) noexcept
{
    Matrix<T, T::shape[0], U::shape[1], T::layout> res{};
    for (std::size_t i = 0; i < T::shape[0]; ++i)
    {
        for (std::size_t j = 0; j < U::shape[1]; ++j)
        {
            for (std::size_t k = 0; k < T::shape[1]; ++k)
                res[i, j] += a[i, k]*b[k, j];
        }
    }

    return res;
}

template <typename T>
[[nodiscard]] constexpr Matrix<T, T::shape[1], T::shape[0], T::layout>
transpose(const T& matrix)
{
    Matrix<T, T::shape[1], T::shape[0], T::layout> res{};
    for (std::size_t i = 0; i < T::shape[0]; ++i)
    {
        for (std::size_t j = 0; j < T::shape[1]; ++j)
            res[j, i] = matrix[i, j];
    }
    return res;
}

template <vector_like T>
[[nodiscard]] constexpr T normalize(const T& a) noexcept
{
    const typename T::value_type norm = length(a);
    if (norm == typename T::value_type{}) return T{};

    return mul((1.0/norm), a);
}

} // namespace zdm
