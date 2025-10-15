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
concept arithmetic = std::is_arithmetic_v<T>;

template <typename T>
concept matrix_like = std::is_arithmetic_v<typename T::value_type>
    && std::same_as<std::remove_const_t<decltype(T::shape)>, std::array<typename T::size_type, 2>>
    && requires (T matrix, typename T::size_type i, T::size_type j)
    {
        { matrix[i, j] } -> std::same_as<std::add_lvalue_reference_t<typename T::value_type>>;
    };

template <typename T>
concept square_matrix_like = matrix_like<T> && (T::shape[0] == T::shape[1]);

template <typename T>
concept vector_like = std::is_arithmetic_v<typename T::value_type>
    && std::same_as<std::remove_const_t<decltype(std::tuple_size_v<T>)>, typename T::size_type>
    && requires (T vector, typename T::size_type i)
    {
        { vector[i] } -> std::same_as<std::add_lvalue_reference_t<typename T::value_type>>;
    };

template <arithmetic T, std::size_t N>
using Vector = std::array<T, N>;

template <
    arithmetic T, std::size_t N, std::size_t M,
    TransformAction action_param = TransformAction::passive,
    MatrixLayout layout_param = MatrixLayout::column_major
>
struct Matrix
{
    using value_type = T;
    using index_type = std::size_t;
    using size_type = std::size_t;
    using transpose_type = Matrix<T, N, M, action_param, layout_param>;

    static constexpr TransformAction action = action_param;
    static constexpr MatrixLayout layout = layout_param;
    static constexpr std::array<size_type, 2> shape = {N, M};

    std::array<T, N*M> array;

    [[nodiscard]] static constexpr Matrix
    identity() noexcept requires (N == M)
    {
        Matrix res{};
        for (std::size_t i = 0; i < N; ++i)
            res[i, i] = 1.0;
        return res;
    }
    
    [[nodiscard]] constexpr bool
    operator==(const Matrix& other) const noexcept
    {
        return array == other.array;
    }

    [[nodiscard]] constexpr T&
    operator[](std::size_t i, std::size_t j) noexcept
    {
        if constexpr (layout == MatrixLayout::row_major)
            return array[M*i + j];
        else
            return array[N*j + i];
    }

    [[nodiscard]] constexpr const T&
    operator[](std::size_t i, std::size_t j) const noexcept
    {
        if constexpr (layout == MatrixLayout::row_major)
            return array[M*i + j];
        else
            return array[N*j + i];
    }
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
[[nodiscard]] constexpr T::value_type dot_(
    const T& a, const T& b, 
    std::index_sequence<idx...>) noexcept
{
    return ((a[idx]*b[idx]) + ...);
}

} // namespace detail

template <vector_like T>
[[nodiscard]] constexpr T::value_type dot(const T& a, const T& b) noexcept
{
    return detail::dot_(a, b, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <vector_like T>
    requires std::floating_point<typename T::value_type>
[[nodiscard]] inline T::value_type length(const T& a) noexcept
{
    // NOTE: This function should be constexpr when clang decides to support
    // constexpr math.
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
    requires std::same_as<typename T::value_type, typename U::value_type>
        && (T::shape[0] != T::shape[1]) && (T::shape[1] == std::tuple_size_v<U>)
[[nodiscard]] constexpr std::array<typename U::value_type, T::shape[0]>
matmul(const T& mat, const U& vec) noexcept
{
    std::array<typename U::value_type, T::shape[0]> res{};
    for (std::size_t i = 0; i < T::shape[0]; ++i)
    {
        for (std::size_t j = 0; j < T::shape[1]; ++j)
            res[i] += mat[i, j]*vec[j];
    }

    return res;
}

template <square_matrix_like T, vector_like U>
    requires std::same_as<typename T::value_type, typename U::value_type>
        && (T::shape[1] == std::tuple_size_v<U>)
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

template <square_matrix_like T>
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

    return res;
}

template <matrix_like T, matrix_like U>
    requires std::same_as<typename T::value_type, typename U::value_type>
        && (T::shape[1] == U::shape[0])
[[nodiscard]] constexpr Matrix<typename T::value_type, T::shape[0], U::shape[1], T::action, T::layout>
matmul(const T& a, const U& b) noexcept
{
    Matrix<typename T::value_type, T::shape[0], U::shape[1], T::action, T::layout> res{};
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

template <matrix_like T>
    requires requires { typename T::transpose_type; }
[[nodiscard]] constexpr T::transpose_type
transpose(const T& matrix) noexcept
{
    typename T::transpose_type res{};
    for (std::size_t i = 0; i < T::shape[0]; ++i)
    {
        for (std::size_t j = 0; j < T::shape[1]; ++j)
            res[j, i] = matrix[i, j];
    }
    return res;
}

template <vector_like T>
    requires std::floating_point<typename T::value_type>
[[nodiscard]] inline T normalize(const T& a) noexcept
{
    const typename T::value_type norm = length(a);
    if (norm == typename T::value_type{}) return T{};

    return mul((1.0/norm), a);
}

template <
    arithmetic T, std::size_t N,
    TransformAction action_param = TransformAction::passive,
    MatrixLayout layout_param = MatrixLayout::column_major
>
class RotationMatrix
{
public:
    using value_type = T;
    using index_type = std::size_t;
    using size_type = std::size_t;
    using transpose_type = RotationMatrix;

    static constexpr TransformAction action = action_param;
    static constexpr MatrixLayout layout = layout_param;
    static constexpr std::array<size_type, 2> shape = {N, N};

    constexpr explicit RotationMatrix() = default;
    constexpr explicit RotationMatrix(std::array<T, N*N> array): m_matrix{array} {}

    [[nodiscard]] static constexpr RotationMatrix
    from_angle(T angle) noexcept requires (N == 2)
    {
        const T cos_angle = std::cos(angle);
        const T sin_angle = std::sin(angle);
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

    template <Axis axis>
    [[nodiscard]] static constexpr RotationMatrix
    coordinate_axis(T angle) noexcept requires (N == 3)
    {
        if constexpr (axis == Axis::x)
            return axis_x(angle);
        if constexpr (axis == Axis::y)
            return axis_y(angle);
        if constexpr (axis == Axis::z)
            return axis_z(angle);
    }

    [[nodiscard]] static constexpr RotationMatrix
    axis(std::array<T, 3> axis, T angle) noexcept requires (N == 3)
    {
        axis = normalize(axis);
        const T x = axis[0];
        const T y = axis[1];
        const T z = axis[2];
        const T xx = axis[0]*axis[0];
        const T yy = axis[1]*axis[1];
        const T zz = axis[2]*axis[2];
        const T xy = axis[0]*axis[1];
        const T xz = axis[0]*axis[2];
        const T yz = axis[1]*axis[2];
        const T cos_angle = std::cos(angle);
        const T sin_angle = std::sin(angle);
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

    template <Axis axis_alpha, Axis axis_beta>
    [[nodiscard]] static constexpr RotationMatrix
    composite_axes(T alpha, T beta) noexcept requires (N == 3)
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

    template <Axis axis_alpha, Axis axis_beta, Axis axis_gamma>
    [[nodiscard]] static constexpr RotationMatrix
    composite_axes(T alpha, T beta, T gamma) noexcept requires (N == 3)
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

    [[nodiscard]] static RotationMatrix
    align_z(const std::array<double, 3>& vector) noexcept requires (N == 3)
    {
        const std::array<double, 3> unit_vec = normalize(vector);
        const double u_xx = unit_vec[0]*unit_vec[0];
        const double u_yy = unit_vec[1]*unit_vec[1];
        const double u_xy = unit_vec[0]*unit_vec[1];
        const double r2 = u_xx + u_yy;
        if (r2 > 0.0)
        {
            const double scale = 1.0/r2;
            const double u_xx_norm = u_xx*scale;
            const double u_yy_norm = u_yy*scale;
            const double r_xx = u_yy_norm + unit_vec[2]*u_xx_norm;
            const double r_yy = u_xx_norm + unit_vec[2]*u_yy_norm;
            const double r_xy = -(1.0 - unit_vec[2])*(u_xy*scale);
            if constexpr (
                    (layout == MatrixLayout::row_major && action == TransformAction::active)
                    || (layout == MatrixLayout::column_major && action == TransformAction::passive))
                return RotationMatrix({
                     r_xx,         r_xy,        -unit_vec[0],
                     r_xy,         r_yy,        -unit_vec[1],
                     unit_vec[0],  unit_vec[1],  unit_vec[2]
                });
            else
                return RotationMatrix({
                     r_xx,         r_xy,         unit_vec[0],
                     r_xy,         r_yy,         unit_vec[1],
                    -unit_vec[0], -unit_vec[1],  unit_vec[2]
                });
        }
        else
            if constexpr (
                    (layout == MatrixLayout::row_major && action == TransformAction::active)
                    || (layout == MatrixLayout::column_major && action == TransformAction::passive))
                return RotationMatrix({
                     unit_vec[2],  0.0,         -unit_vec[0],
                     0.0,          1.0,         -unit_vec[1],
                     unit_vec[0],  unit_vec[1],  unit_vec[2],
                });
            else
                return RotationMatrix({
                     unit_vec[2],  0.0,          unit_vec[0],
                     0.0,          1.0,          unit_vec[1],
                    -unit_vec[0], -unit_vec[1],  unit_vec[2],
                });
    }

    [[nodiscard]] static RotationMatrix
    align_z_inverse(const std::array<double, 3>& vector) noexcept requires (N == 3)
    {
        const std::array<double, 3> unit_vec = normalize(vector);
        const double u_xx = unit_vec[0]*unit_vec[0];
        const double u_yy = unit_vec[1]*unit_vec[1];
        const double u_xy = unit_vec[0]*unit_vec[1];
        const double r2 = u_xx + u_yy;
        if (r2 > 0.0)
        {
            const double scale = 1.0/r2;
            const double u_xx_norm = u_xx*scale;
            const double u_yy_norm = u_yy*scale;
            const double r_xx = u_yy_norm + unit_vec[2]*u_xx_norm;
            const double r_yy = u_xx_norm + unit_vec[2]*u_yy_norm;
            const double r_xy = -(1.0 - unit_vec[2])*(u_xy*scale);
            if constexpr (
                    (layout == MatrixLayout::row_major && action == TransformAction::active)
                    || (layout == MatrixLayout::column_major && action == TransformAction::passive))
                return RotationMatrix({
                     r_xx,         r_xy,         unit_vec[0],
                     r_xy,         r_yy,         unit_vec[1],
                    -unit_vec[0], -unit_vec[1],  unit_vec[2]
                });
            else
                return RotationMatrix({
                     r_xx,         r_xy,        -unit_vec[0],
                     r_xy,         r_yy,        -unit_vec[1],
                     unit_vec[0],  unit_vec[1],  unit_vec[2]
                });
        }
        else
            if constexpr (
                    (layout == MatrixLayout::row_major && action == TransformAction::active)
                    || (layout == MatrixLayout::column_major && action == TransformAction::passive))
                return RotationMatrix({
                     unit_vec[2],  0.0,          unit_vec[0],
                     0.0,          1.0,          unit_vec[1],
                    -unit_vec[0], -unit_vec[1],  unit_vec[2],
                });
            else
                return RotationMatrix({
                     unit_vec[2],  0.0,         -unit_vec[0],
                     0.0,          1.0,         -unit_vec[1],
                     unit_vec[0],  unit_vec[1],  unit_vec[2],
                });
    }

    [[nodiscard]] constexpr operator Matrix<T, N, N, action, layout>() noexcept { return m_matrix; }
    [[nodiscard]] constexpr operator std::array<T, N*N>() noexcept { return m_matrix.array; }

    [[nodiscard]] constexpr T&
    operator[](std::size_t i, std::size_t j) noexcept { return m_matrix[i, j]; }

    [[nodiscard]] constexpr const T&
    operator[](std::size_t i, std::size_t j) const noexcept { return m_matrix[i, j]; }

    [[nodiscard]] constexpr RotationMatrix
    inverse() const noexcept
    {
        return transpose(*m_matrix);
    }

private:
    [[nodiscard]] static constexpr RotationMatrix axis_x(T angle) noexcept requires (N == 3)
    {
        const T ca = std::cos(angle);
        const T sa = std::sin(angle);
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

    [[nodiscard]] static constexpr RotationMatrix axis_y(T angle) noexcept requires (N == 3)
    {
        const T ca = std::cos(angle);
        const T sa = std::sin(angle);
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

    [[nodiscard]] static constexpr RotationMatrix axis_z(T angle) noexcept requires (N == 3)
    {
        const T ca = std::cos(angle);
        const T sa = std::sin(angle);
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

    [[nodiscard]] static constexpr RotationMatrix
    composite_axes_xy(T alpha, T beta) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
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

    [[nodiscard]] static constexpr RotationMatrix
    composite_axes_xz(T alpha, T beta) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
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

    [[nodiscard]] static constexpr RotationMatrix
    composite_axes_yx(T alpha, T beta) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
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

    [[nodiscard]] static constexpr RotationMatrix
    composite_axes_yz(T alpha, T beta) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
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

    [[nodiscard]] static constexpr RotationMatrix
    composite_axes_zx(T alpha, T beta) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
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

    [[nodiscard]] static constexpr RotationMatrix
    composite_axes_zy(T alpha, T beta) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
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

    [[nodiscard]] static constexpr RotationMatrix
    composite_axes_xyx(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos(gamma);
        const T sg = std::sin(gamma);
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

    [[nodiscard]] static constexpr RotationMatrix
    composite_axes_xzx(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos(gamma);
        const T sg = std::sin(gamma);
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

    [[nodiscard]] static constexpr RotationMatrix
    composite_axes_yxy(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos(gamma);
        const T sg = std::sin(gamma);
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

    [[nodiscard]] static constexpr RotationMatrix
    composite_axes_yzy(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos(gamma);
        const T sg = std::sin(gamma);
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

    [[nodiscard]] static constexpr RotationMatrix
    composite_axes_zxz(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos(gamma);
        const T sg = std::sin(gamma);
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

    [[nodiscard]] static constexpr RotationMatrix
    composite_axes_zyz(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos(gamma);
        const T sg = std::sin(gamma);
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

    [[nodiscard]] static constexpr RotationMatrix
    composite_axes_xyz(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos(gamma);
        const T sg = std::sin(gamma);
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

    [[nodiscard]] static constexpr RotationMatrix
    composite_axes_xzy(T alpha, T beta, T gamma) noexcept requires(N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos(gamma);
        const T sg = std::sin(gamma);
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

    [[nodiscard]] static constexpr RotationMatrix
    composite_axes_yxz(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos(gamma);
        const T sg = std::sin(gamma);
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

    [[nodiscard]] static constexpr RotationMatrix
    composite_axes_yzx(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos(gamma);
        const T sg = std::sin(gamma);
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

    [[nodiscard]] static constexpr RotationMatrix
    composite_axes_zxy(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos(gamma);
        const T sg = std::sin(gamma);
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

    [[nodiscard]] static constexpr RotationMatrix
    composite_axes_zyx(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos(gamma);
        const T sg = std::sin(gamma);
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
    arithmetic T, std::size_t N,
    TransformAction action_param = TransformAction::passive,
    MatrixLayout layout_param = MatrixLayout::column_major>
class RigidMatrix
{
public:
    using value_type = T;
    using index_type = std::size_t;
    using size_type = std::size_t;
    using transpose_type = Matrix<T, N + 1, N + 1, action_param, layout_param>;

    static constexpr TransformAction action = action_param;
    static constexpr MatrixLayout layout = layout_param;
    static constexpr std::array<size_type, 2> shape = {N + 1, N + 1};

    constexpr RigidMatrix() = default;
    constexpr explicit RigidMatrix(std::array<T, (N + 1)*(N + 1)> array): m_matrix{array} {}

    [[nodiscard]] static constexpr RigidMatrix
    from(RotationMatrix<T, N, action, layout> rotation, std::array<T, N> translation) noexcept
    {
        RigidMatrix res;
        for (std::size_t i = 0; i < N; ++i)
        {
            for (std::size_t j = 0; j < N; ++j)
                res[i, j] = rotation[i, j];
            res[i, N] = translation[i];
        }
        res[N, N] = 1.0;
        return res;
    }

    [[nodiscard]] static constexpr RigidMatrix
    from(RotationMatrix<T, N, action, layout> rotation) noexcept
    {
        RigidMatrix res;
        for (std::size_t i = 0; i < N; ++i)
        {
            for (std::size_t j = 0; j < N; ++j)
                res[i, j] = rotation[i, j];
        }
        res[N, N] = 1.0;
        return res;
    }

    [[nodiscard]] static constexpr RigidMatrix
    from(std::array<T, N> translation) noexcept
    {
        RigidMatrix res;
        for (std::size_t i = 0; i < N; ++i)
            res[i, N] = translation[i];
        res[N, N] = 1.0;
        return res;
    }

    [[nodiscard]] constexpr T&
    operator[](std::size_t i, std::size_t j) noexcept { return m_matrix[i, j]; }

    [[nodiscard]] constexpr const T&
    operator[](std::size_t i, std::size_t j) const noexcept { return m_matrix[i, j]; }

    [[nodiscard]] constexpr RotationMatrix<value_type, N, action, layout>
    extract_rotation() const noexcept
    {
        RotationMatrix<value_type, N, action, layout> res{};
        for (std::size_t i = 0; i < N; ++i)
        {
            for (std::size_t j = 0; j < N; ++j)
                res[i, j] = m_matrix[i, j];
        }
        return res;
    }

    [[nodiscard]] constexpr std::array<value_type, N>
    extract_translation() const noexcept
    {
        std::array<value_type, N> res{};
        for (std::size_t i = 0; i < N; ++i)
            res[i] = m_matrix[i, N];
        return res;
    }

    [[nodiscard]] constexpr RigidMatrix
    inverse() const noexcept
    {
        const auto inverse_rotation = extract_rotation().inverse();
        return RigidMatrix::from(inverse_rotation, matmul(inverse_rotation, extract_translation()));
    }

private:
    Matrix<T, N + 1, N + 1, action, layout> m_matrix = Matrix<T, N, N, action, layout>::identity();
};

template <
    arithmetic T, std::size_t N,
    TransformAction action_param = TransformAction::passive,
    MatrixLayout matrix_layout_param = MatrixLayout::column_major
>
class RigidTransform
{
    using value_type = T;
    using size_type = std::size_t;
    using vector_type = std::array<T, N>;
    using matrix_type = Matrix<T, N, N, action_param, matrix_layout_param>;

    static constexpr TransformAction action = action_param;
    static constexpr MatrixLayout matrix_layout = matrix_layout_param;

    constexpr RigidTransform() = default;
    constexpr explicit RigidTransform(
        RotationMatrix<T, N, action, matrix_layout> rotation, std::array<T, N> translation):
        m_rotation{rotation}, m_translation{translation} {}

    [[nodiscard]] constexpr vector_type operator()(const vector_type& vector) const noexcept
    {
        return matmul(m_rotation, vector) + m_translation;
    }

    template <matrix_like M>
    [[nodiscard]] constexpr matrix_type operator()(const M& matrix) const noexcept
    {
        return matmul(m_rotation, matmul(matrix, transpose(m_rotation)));
    }

    [[nodiscard]] constexpr const RotationMatrix<T, N, action, matrix_layout>&
    rotation_matrix() const noexcept { return m_rotation; }

    [[nodiscard]] constexpr const std::array<T, N>&
    translation() const noexcept { return m_translation; }

private:
    RotationMatrix<T, N, action, matrix_layout> m_rotation{};
    std::array<T, N> m_translation;
};


} // namespace zdm
