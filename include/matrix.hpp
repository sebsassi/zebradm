#pragma once

#include <array>
#include <cmath>

#include "concepts.hpp"
#include "linalg.hpp"
#include "vector.hpp"

namespace zdm
{

namespace la
{

enum class MatrixLayout
{
    row_major,
    column_major
};

enum class Action
{
    active,
    passive
};

enum class Chaining
{
    intrinsic,
    extrinsic
};

enum class EulerConvention
{
    xzx,
    xyx,
    yxy,
    yzy,
    zyz,
    zxz
};

enum class TaitBryanConvention
{
    xyz,
    xzy,
    yxz,
    yzx,
    zxy,
    zyx
};

template <
    arithmetic T, std::size_t N, std::size_t M,
    Action action_param = Action::passive,
    MatrixLayout layout_param = MatrixLayout::column_major
>
struct Matrix
{
    using value_type = T;
    using index_type = std::size_t;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using transpose_type = Matrix<T, M, N, action_param, layout_param>;

    static constexpr Action action = action_param;
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

    [[nodiscard]] explicit constexpr 
    operator std::array<T, N*M>() const noexcept { return array; }

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

    [[nodiscard]] constexpr Matrix
    operator*(const Matrix& other) const noexcept requires (N == M) { return matmul(*this, other); }

    template <vector_like V>
    [[nodiscard]] constexpr Vector<T, N>
    operator*(const V& vector) const noexcept { return matmul(*this, vector); }
};

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

template <
    std::floating_point T, std::size_t N,
    Action action_param = Action::passive,
    MatrixLayout layout_param = MatrixLayout::column_major
>
class RotationMatrix
{
public:
    using value_type = T;
    using index_type = std::size_t;
    using size_type = std::size_t;
    using transpose_type = RotationMatrix;

    static constexpr Action action = action_param;
    static constexpr MatrixLayout layout = layout_param;
    static constexpr std::array<size_type, 2> shape = {N, N};

    constexpr explicit RotationMatrix() = default;
    constexpr explicit RotationMatrix(std::array<T, N*N> array): m_matrix{array} {}

    [[nodiscard]] static constexpr RotationMatrix
    identity() noexcept
    {
        RotationMatrix res{};
        for (std::size_t i = 0; i < N; ++i)
            res[i, i] = 1.0;
        return res;
    }

    [[nodiscard]] static constexpr RotationMatrix
    from_angle(T angle) noexcept requires (N == 2)
    {
        const T cos_angle = std::cos(angle);
        const T sin_angle = std::sin(angle);
        if constexpr (
                (layout == MatrixLayout::row_major && action == Action::active)
                || (layout == MatrixLayout::column_major && action == Action::passive))
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
                (layout == MatrixLayout::row_major && action == Action::active)
                || (layout == MatrixLayout::column_major && action == Action::passive))
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

    // Genuine product of axis rotation matrices.
    template <Axis axis_alpha, Axis axis_beta>
    [[nodiscard]] static constexpr RotationMatrix
    product_axes(T alpha, T beta) noexcept requires (N == 3)
    {
        if constexpr (axis_alpha == Axis::x && axis_beta == Axis::y)
            return product_axes_xy<Order::keep>(alpha, beta);
        if constexpr (axis_alpha == Axis::x && axis_beta == Axis::z)
            return product_axes_xz<Order::keep>(alpha, beta);
        if constexpr (axis_alpha == Axis::y && axis_beta == Axis::x)
            return product_axes_xy<Order::reverse>(alpha, beta);
        if constexpr (axis_alpha == Axis::y && axis_beta == Axis::z)
            return product_axes_yz<Order::keep>(alpha, beta);
        if constexpr (axis_alpha == Axis::z && axis_beta == Axis::x)
            return product_axes_xz<Order::reverse>(alpha, beta);
        if constexpr (axis_alpha == Axis::z && axis_beta == Axis::y)
            return product_axes_yz<Order::reverse>(alpha, beta);
    }

    template <Axis axis_alpha, Axis axis_beta, Chaining chaining>
    [[nodiscard]] static constexpr RotationMatrix
    composite_axes(T alpha, T beta) noexcept requires (N == 3)
    {
        // Let `alpha` be the first rotation and `beta` be the second rotation
        // in a chain, and `i` be the axis of the first rotation and `j` be the
        // axis of the second rotation. The rotation matrix is then formed as
        // follows
        //      extrinsic, active: `R_j(beta)R_i(alpha)`
        //      extrinsic, passive: `R_i(alpha)R_j(beta)`
        //      intrinsic, active: `R_j(beta)R_i(alpha)`
        //      intrinsic, passive: `R_i(alpha)R_j(beta)`
        // where `R` is the respective active/passive rotation matrix.
        constexpr bool commute
            = ((chaining == Chaining::extrinsic && action == Action::active)
            || (chaining == Chaining::intrinsic && action == Action::passive));
        constexpr Order order = (commute) ? Order::reverse : Order::keep;
        constexpr Order reverse_order = (order == Order::keep) ? Order::reverse : Order::keep;
        if constexpr (axis_alpha == Axis::x && axis_beta == Axis::y)
            return product_axes_xy<order>((commute) ? beta : alpha, (commute) ? alpha : beta);
        if constexpr (axis_alpha == Axis::x && axis_beta == Axis::z)
            return product_axes_xz<order>((commute) ? beta : alpha, (commute) ? alpha : beta);
        if constexpr (axis_alpha == Axis::y && axis_beta == Axis::x)
            return product_axes_xy<reverse_order>((commute) ? beta : alpha, (commute) ? alpha : beta);
        if constexpr (axis_alpha == Axis::y && axis_beta == Axis::z)
            return product_axes_yz<order>((commute) ? beta : alpha, (commute) ? alpha : beta);
        if constexpr (axis_alpha == Axis::z && axis_beta == Axis::x)
            return product_axes_xz<reverse_order>((commute) ? beta : alpha, (commute) ? alpha : beta);
        if constexpr (axis_alpha == Axis::z && axis_beta == Axis::y)
            return product_axes_yz<reverse_order>((commute) ? beta : alpha, (commute) ? alpha : beta);
}

    // Genuine product of axis rotation matrices.
    template <Axis axis_alpha, Axis axis_beta, Axis axis_gamma>
    [[nodiscard]] static constexpr RotationMatrix
    product_axes(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        if constexpr (axis_alpha == Axis::x && axis_beta == Axis::y && axis_gamma == Axis::x)
            return product_axes_xyx(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::x && axis_beta == Axis::z && axis_gamma == Axis::x)
            return product_axes_xzx(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::y && axis_beta == Axis::x && axis_gamma == Axis::y)
            return product_axes_yxy(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::y && axis_beta == Axis::z && axis_gamma == Axis::y)
            return product_axes_yzy(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::z && axis_beta == Axis::x && axis_gamma == Axis::z)
            return product_axes_zxz(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::z && axis_beta == Axis::y && axis_gamma == Axis::z)
            return product_axes_zyz(alpha, beta, gamma);

        if constexpr (axis_alpha == Axis::x && axis_beta == Axis::y && axis_gamma == Axis::z)
            return product_axes_xyz<Order::keep>(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::x && axis_beta == Axis::z && axis_gamma == Axis::y)
            return product_axes_xzy<Order::keep>(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::y && axis_beta == Axis::x && axis_gamma == Axis::z)
            return product_axes_yxz<Order::keep>(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::y && axis_beta == Axis::z && axis_gamma == Axis::x)
            return product_axes_xzy<Order::reverse>(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::z && axis_beta == Axis::x && axis_gamma == Axis::y)
            return product_axes_yxz<Order::reverse>(alpha, beta, gamma);
        if constexpr (axis_alpha == Axis::z && axis_beta == Axis::y && axis_gamma == Axis::x)
            return product_axes_xyz<Order::reverse>(alpha, beta, gamma);
    }

    template <EulerConvention convention, Chaining chaining>
    [[nodiscard]] static constexpr RotationMatrix
    from_euler_angles(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        constexpr bool commute
            = ((chaining == Chaining::extrinsic && action == Action::active)
            || (chaining == Chaining::intrinsic && action == Action::passive));
        if constexpr (convention == EulerConvention::xyx)
            return product_axes_xyx((commute) ? gamma : alpha, beta, (commute) ? alpha: gamma);
        if constexpr (convention == EulerConvention::xzx)
            return product_axes_xzx((commute) ? gamma : alpha, beta, (commute) ? alpha: gamma);
        if constexpr (convention == EulerConvention::yxy)
            return product_axes_yxy((commute) ? gamma : alpha, beta, (commute) ? alpha: gamma);
        if constexpr (convention == EulerConvention::yzy)
            return product_axes_yzy((commute) ? gamma : alpha, beta, (commute) ? alpha: gamma);
        if constexpr (convention == EulerConvention::zxz)
            return product_axes_zxz((commute) ? gamma : alpha, beta, (commute) ? alpha: gamma);
        if constexpr (convention == EulerConvention::zyz)
            return product_axes_zyz((commute) ? gamma : alpha, beta, (commute) ? alpha: gamma);
    }

    template <TaitBryanConvention convention, Chaining chaining>
    [[nodiscard]] static constexpr RotationMatrix
    from_tait_bryan_angles(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        constexpr bool commute
            = ((chaining == Chaining::extrinsic && action == Action::active)
            || (chaining == Chaining::intrinsic && action == Action::passive));
        constexpr Order order = (commute) ? Order::reverse : Order::keep;
        constexpr Order reverse_order = (order == Order::keep) ? Order::reverse : Order::keep;
        if constexpr (convention == TaitBryanConvention::xyz)
            return product_axes_xyz<order>((commute) ? gamma : alpha, beta, (commute) ? alpha : gamma);
        if constexpr (convention == TaitBryanConvention::xzy)
            return product_axes_xzy<order>((commute) ? gamma : alpha, beta, (commute) ? alpha : gamma);
        if constexpr (convention == TaitBryanConvention::yxz)
            return product_axes_yxz<order>((commute) ? gamma : alpha, beta, (commute) ? alpha : gamma);
        if constexpr (convention == TaitBryanConvention::yzx)
            return product_axes_xzy<reverse_order>((commute) ? gamma : alpha, beta, (commute) ? alpha : gamma);
        if constexpr (convention == TaitBryanConvention::zxy)
            return product_axes_yxz<reverse_order>((commute) ? gamma : alpha, beta, (commute) ? alpha : gamma);
        if constexpr (convention == TaitBryanConvention::zyx)
            return product_axes_xyz<reverse_order>((commute) ? gamma : alpha, beta, (commute) ? alpha : gamma);
    }

    [[nodiscard]] static RotationMatrix
    align_z(const Vector<double, 3>& vector) noexcept requires (N == 3)
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
                    (layout == MatrixLayout::row_major && action == Action::active)
                    || (layout == MatrixLayout::column_major && action == Action::passive))
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
        else [[unlikely]]
            if constexpr (
                    (layout == MatrixLayout::row_major && action == Action::active)
                    || (layout == MatrixLayout::column_major && action == Action::passive))
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

    [[nodiscard]] static RotationMatrix
    align_z_inverse(const Vector<double, 3>& vector) noexcept requires (N == 3)
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
                    (layout == MatrixLayout::row_major && action == Action::active)
                    || (layout == MatrixLayout::column_major && action == Action::passive))
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
        else [[unlikely]]
            if constexpr (
                    (layout == MatrixLayout::row_major && action == Action::active)
                    || (layout == MatrixLayout::column_major && action == Action::passive))
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

    [[nodiscard]] constexpr operator Matrix<T, N, N, action, layout>() const noexcept { return m_matrix; }
    [[nodiscard]] constexpr explicit operator std::array<T, N*N>() const noexcept { return m_matrix.array; }

    [[nodiscard]] constexpr bool
    operator==(const RotationMatrix& other) const noexcept
    {
        return m_matrix.array == std::array<T, N*N>(other);
    }

    [[nodiscard]] constexpr T&
    operator[](std::size_t i, std::size_t j) noexcept { return m_matrix[i, j]; }

    [[nodiscard]] constexpr const T&
    operator[](std::size_t i, std::size_t j) const noexcept { return m_matrix[i, j]; }

    [[nodiscard]] constexpr RotationMatrix
    operator*(const RotationMatrix& other) const noexcept { return matmul(*this, other); }

    [[nodiscard]] friend constexpr Matrix<T, N, N, action, layout>
    operator*(const RotationMatrix& r, const Matrix<T, N, N, action, layout>& m) { return matmul(r, m); }

    [[nodiscard]] friend constexpr Matrix<T, N, N, action, layout>
    operator*(const Matrix<T, N, N, action, layout>& m, const RotationMatrix& r) { return matmul(m, r); }

    [[nodiscard]] constexpr Vector<T, N>
    operator*(const Vector<T, N>& vector) const noexcept { return matmul(*this, vector); }

    template <vector_like V>
    [[nodiscard]] constexpr V
    operator*(const V& vector) const noexcept { return matmul(*this, vector); }

    [[nodiscard]] constexpr RotationMatrix
    inverse() const noexcept
    {
        return transpose(*m_matrix);
    }

private:
    enum class Order { keep, reverse };

    [[nodiscard]] static constexpr RotationMatrix axis_x(T angle) noexcept requires (N == 3)
    {
        const T ca = std::cos(angle);
        const T sa = std::sin(angle);
        if constexpr (
                (layout == MatrixLayout::row_major && action == Action::active)
                || (layout == MatrixLayout::column_major && action == Action::passive))
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
                (layout == MatrixLayout::row_major && action == Action::active)
                || (layout == MatrixLayout::column_major && action == Action::passive))
            return RotationMatrix({
                 ca,   0.0,  sa,
                 0.0,  1.0,  0.0,
                -sa,   0.0,  ca
            });
        else
            return RotationMatrix({
                 ca,   0.0, -sa,
                 0.0,  1.0,  0.0,
                 sa,   0.0,  ca
            });
    }

    [[nodiscard]] static constexpr RotationMatrix axis_z(T angle) noexcept requires (N == 3)
    {
        const T ca = std::cos(angle);
        const T sa = std::sin(angle);
        if constexpr (
                (layout == MatrixLayout::row_major && action == Action::active)
                || (layout == MatrixLayout::column_major && action == Action::passive))
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

    template <Order order>
    [[nodiscard]] static constexpr RotationMatrix
    product_axes_xy(T alpha, T beta) noexcept requires (N == 3)
    {
        const T ca = std::cos((order == Order::reverse) ? beta : alpha);
        const T sa = std::sin((order == Order::reverse) ? beta : alpha);
        const T cb = std::cos((order == Order::reverse) ? alpha : beta);
        const T sb = std::sin((order == Order::reverse) ? alpha : beta);
        if constexpr (
                (layout == MatrixLayout::row_major && action == Action::passive && order == Order::keep)
                || (layout == MatrixLayout::column_major && action == Action::active && order == Order::reverse))
            return RotationMatrix({
                 cb,     0.0,   -sb,
                 sa*sb,  ca,     cb*sa,
                 ca*sb, -sa,     ca*cb
            });
        else if constexpr (
                (layout == MatrixLayout::column_major && action == Action::passive && order == Order::keep)
                || (layout == MatrixLayout::row_major && action == Action::active && order == Order::reverse))
            return RotationMatrix({
                 cb,     sa*sb,  ca*sb,
                 0.0,    ca,    -sa,
                -sb,     cb*sa,  ca*cb
            });
        else if constexpr (
                (layout == MatrixLayout::row_major && action == Action::active && order == Order::keep)
                || (layout == MatrixLayout::column_major && action == Action::passive && order == Order::reverse))
            return RotationMatrix({
                 cb,     0.0,    sb,
                 sa*sb,  ca,    -cb*sa,
                -ca*sb,  sa,     ca*cb
            });
        else if constexpr (
                (layout == MatrixLayout::column_major && action == Action::active && order == Order::keep)
                || (layout == MatrixLayout::row_major && action == Action::passive && order == Order::reverse))
            return RotationMatrix({
                 cb,     sa*sb, -ca*sb,
                 0.0,    ca,     sa,
                 sb,    -cb*sa,  ca*cb
            });
    }

    template <Order order>
    [[nodiscard]] static constexpr RotationMatrix
    product_axes_xz(T alpha, T beta) noexcept requires (N == 3)
    {
        const T ca = std::cos((order == Order::reverse) ? beta : alpha);
        const T sa = std::sin((order == Order::reverse) ? beta : alpha);
        const T cb = std::cos((order == Order::reverse) ? alpha : beta);
        const T sb = std::sin((order == Order::reverse) ? alpha : beta);
        if constexpr (
                (layout == MatrixLayout::row_major && action == Action::passive && order == Order::keep)
                || (layout == MatrixLayout::column_major && action == Action::active && order == Order::reverse))
            return RotationMatrix({
                 cb,     sb,     0.0,
                -ca*sb,  ca*cb,  sa,
                 sa*sb, -cb*sa,  ca
            });
        else if constexpr (
                (layout == MatrixLayout::column_major && action == Action::passive && order == Order::keep)
                || (layout == MatrixLayout::row_major && action == Action::active && order == Order::reverse))
            return RotationMatrix({
                 cb,    -ca*sb,  sa*sb,
                 sb,     ca*cb, -cb*sa,
                 0.0,    sa,     ca
            });
        else if constexpr (
                (layout == MatrixLayout::row_major && action == Action::active && order == Order::keep)
                || (layout == MatrixLayout::column_major && action == Action::passive && order == Order::reverse))
            return RotationMatrix({
                 cb,    -sb,     0.0,
                 ca*sb,  ca*cb, -sa,
                 sa*sb,  cb*sa,  ca
            });
        else if constexpr (
                (layout == MatrixLayout::column_major && action == Action::active && order == Order::keep)
                || (layout == MatrixLayout::row_major && action == Action::passive && order == Order::reverse))
            return RotationMatrix({
                 cb,     ca*sb,  sa*sb,
                -sb,     ca*cb,  cb*sa,
                 0.0,   -sa,     ca
            });
    }

    template <Order order>
    [[nodiscard]] static constexpr RotationMatrix
    product_axes_yz(T alpha, T beta) noexcept requires (N == 3)
    {
        const T ca = std::cos((order == Order::reverse) ? beta : alpha);
        const T sa = std::sin((order == Order::reverse) ? beta : alpha);
        const T cb = std::cos((order == Order::reverse) ? alpha : beta);
        const T sb = std::sin((order == Order::reverse) ? alpha : beta);
        if constexpr (
                (layout == MatrixLayout::row_major && action == Action::passive && order == Order::keep)
                || (layout == MatrixLayout::column_major && action == Action::active && order == Order::reverse))
            return RotationMatrix({
                 ca*cb,  ca*sb, -sa,
                -sb,     cb,     0.0,
                 cb*sa,  sa*sb,  ca
            });
        else if constexpr (
                (layout == MatrixLayout::column_major && action == Action::passive && order == Order::keep)
                || (layout == MatrixLayout::row_major && action == Action::active && order == Order::reverse))
            return RotationMatrix({
                 ca*cb, -sb,     cb*sa,
                 ca*sb,  cb,     sa*sb,
                -sa,     0.0,    ca
            });
        else if constexpr (
                (layout == MatrixLayout::row_major && action == Action::active && order == Order::keep)
                || (layout == MatrixLayout::column_major && action == Action::passive && order == Order::reverse))
            return RotationMatrix({
                 ca*cb, -ca*sb,  sa,
                 sb,     cb,     0.0,
                -cb*sa,  sa*sb,  ca
            });
        else if constexpr (
                (layout == MatrixLayout::column_major && action == Action::active && order == Order::keep)
                || (layout == MatrixLayout::row_major && action == Action::passive && order == Order::reverse))
            return RotationMatrix({
                 ca*cb,  sb,    -cb*sa,
                -ca*sb,  cb,     sa*sb,
                 sa,     0.0,    ca
            });
    }

    [[nodiscard]] static constexpr RotationMatrix
    product_axes_xyx(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos(gamma);
        const T sg = std::sin(gamma);
        if constexpr (layout == MatrixLayout::row_major && action == Action::passive)
            return RotationMatrix({
                 cb,                sb*sg,            -cg*sb,
                 sa*sb,             ca*cg - cb*sa*sg,  ca*sg + cb*cg*sa,
                 ca*sb,            -cg*sa - ca*cb*sg, -sa*sg + ca*cb*cg 
            });
        else if constexpr (layout == MatrixLayout::column_major && action == Action::passive)
            return RotationMatrix({
                 cb,                sa*sb,             ca*sb,
                 sb*sg,             ca*cg - cb*sa*sg, -cg*sa - ca*cb*sg,
                -cg*sb,             ca*sg + cb*cg*sa, -sa*sg + ca*cb*cg
            });
        else if constexpr (layout == MatrixLayout::row_major && action == Action::active)
            return RotationMatrix({
                 cb,                sb*sg,             cg*sb,
                 sa*sb,             ca*cg - cb*sa*sg, -ca*sg - cb*cg*sa,
                -ca*sb,             cg*sa + ca*cb*sg, -sa*sg + ca*cb*cg 
            });
        else if constexpr (layout == MatrixLayout::column_major && action == Action::active)
            return RotationMatrix({
                 cb,                sa*sb,            -ca*sb,
                 sb*sg,             ca*cg - cb*sa*sg,  cg*sa + ca*cb*sg,
                 cg*sb,            -ca*sg - cb*cg*sa, -sa*sg + ca*cb*cg
            });
    }

    [[nodiscard]] static constexpr RotationMatrix
    product_axes_xzx(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos(gamma);
        const T sg = std::sin(gamma);
        if constexpr (layout == MatrixLayout::row_major && action == Action::passive)
            return RotationMatrix({
                 cb,                cg*sb,             sb*sg,
                -ca*sb,            -sa*sg + ca*cb*cg,  cg*sa + ca*cb*sg,
                 sa*sb,            -ca*sg - cb*cg*sa,  ca*cg - cb*sa*sg
            });
        else if constexpr (layout == MatrixLayout::column_major && action == Action::passive)
            return RotationMatrix({
                 cb,               -ca*sb,             sa*sb,
                 cg*sb,            -sa*sg + ca*cb*cg, -ca*sg - cb*cg*sa,
                 sb*sg,             cg*sa + ca*cb*sg,  ca*cg - cb*sa*sg
            });
        if constexpr (layout == MatrixLayout::row_major && action == Action::active)
            return RotationMatrix({
                 cb,               -cg*sb,             sb*sg,
                 ca*sb,             ca*cb*cg - sa*sg, -cg*sa - ca*cb*sg,
                 sa*sb,             ca*sg + cb*cg*sa,  ca*cg - cb*sa*sg
            });
        else if constexpr (layout == MatrixLayout::column_major && action == Action::active)
            return RotationMatrix({
                 cb,                ca*sb,             sa*sb,
                -cg*sb,             ca*cb*cg - sa*sg,  ca*sg + cb*cg*sa,
                 sb*sg,            -cg*sa - ca*cb*sg,  ca*cg - cb*sa*sg
            });
    }

    [[nodiscard]] static constexpr RotationMatrix
    product_axes_yxy(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos(gamma);
        const T sg = std::sin(gamma);
        if constexpr (layout == MatrixLayout::row_major && action == Action::passive)
            return RotationMatrix({
                 ca*cg - cb*sa*sg,  sa*sb,            -ca*sg - cb*cg*sa,
                 sb*sg,             cb,                cg*sb,
                 cg*sa + ca*cb*sg, -ca*sb,            -sa*sg + ca*cb*cg
            });
        else if constexpr (layout == MatrixLayout::column_major && action == Action::passive)
            return RotationMatrix({
                 ca*cg - cb*sa*sg,  sb*sg,             cg*sa + ca*cb*sg,
                 sa*sb,             cb,               -ca*sb,
                -ca*sg - cb*cg*sa,  cg*sb,            -sa*sg + ca*cb*cg
            });
        if constexpr (layout == MatrixLayout::row_major && action == Action::active)
            return RotationMatrix({
                 ca*cg - cb*sa*sg,  sa*sb,             ca*sg + cb*cg*sa,
                 sb*sg,             cb,               -cg*sb,
                -cg*sa - ca*cb*sg,  ca*sb,            -sa*sg + ca*cb*cg
            });
        else if constexpr (layout == MatrixLayout::column_major && action == Action::active)
            return RotationMatrix({
                 ca*cg - cb*sa*sg,  sb*sg,            -cg*sa - ca*cb*sg,
                 sa*sb,             cb,                ca*sb,
                 ca*sg + cb*cg*sa, -cg*sb,            -sa*sg + ca*cb*cg
            });
    }

    [[nodiscard]] static constexpr RotationMatrix
    product_axes_yzy(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos(gamma);
        const T sg = std::sin(gamma);
        if constexpr (layout == MatrixLayout::row_major && action == Action::passive)
            return RotationMatrix({
                -sa*sg + ca*cb*cg,  ca*sb,            -cg*sa - ca*cb*sg,
                -cg*sb,             cb,                sb*sg,
                 ca*sg + cb*cg*sa,  sa*sb,             ca*cg - cb*sa*sg
            });
        else if constexpr (layout == MatrixLayout::column_major && action == Action::passive)
            return RotationMatrix({
                -sa*sg + ca*cb*cg, -cg*sb,             ca*sg + cb*cg*sa,
                 ca*sb,             cb,                sa*sb,
                -cg*sa - ca*cb*sg,  sb*sg,             ca*cg - cb*sa*sg
            });
        if constexpr (layout == MatrixLayout::row_major && action == Action::active)
            return RotationMatrix({
                -sa*sg + ca*cb*cg, -ca*sb,             cg*sa + ca*cb*sg,
                 cg*sb,             cb,                sb*sg,
                -ca*sg - cb*cg*sa,  sa*sb,             ca*cg - cb*sa*sg
            });
        else if constexpr (layout == MatrixLayout::column_major && action == Action::active)
            return RotationMatrix({
                -sa*sg + ca*cb*cg,  cg*sb,            -ca*sg - cb*cg*sa,
                -ca*sb,             cb,                sa*sb,
                 cg*sa + ca*cb*sg,  sb*sg,             ca*cg - cb*sa*sg
            });
    }

    [[nodiscard]] static constexpr RotationMatrix
    product_axes_zxz(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos(gamma);
        const T sg = std::sin(gamma);
        if constexpr (layout == MatrixLayout::row_major && action == Action::passive)
            return RotationMatrix({
                 ca*cg - cb*sa*sg,  ca*sg + cb*cg*sa,  sa*sb,
                -cg*sa - ca*cb*sg, -sa*sg + ca*cb*cg,  ca*sb,
                 sb*sg,            -cg*sb,             cb
            });
        else if constexpr (layout == MatrixLayout::column_major && action == Action::passive)
            return RotationMatrix({
                 ca*cg - cb*sa*sg, -cg*sa - ca*cb*sg,  sb*sg,
                 ca*sg + cb*cg*sa, -sa*sg + ca*cb*cg, -cg*sb,
                 sa*sb,             ca*sb,             cb
            });
        if constexpr (layout == MatrixLayout::row_major && action == Action::active)
            return RotationMatrix({
                 ca*cg - cb*sa*sg, -ca*sg - cb*cg*sa,  sa*sb,
                 cg*sa + ca*cb*sg, -sa*sg + ca*cb*cg, -ca*sb,
                 sb*sg,             cg*sb,             cb
            });
        else if constexpr (layout == MatrixLayout::column_major && action == Action::active)
            return RotationMatrix({
                 ca*cg - cb*sa*sg,  cg*sa + ca*cb*sg,  sb*sg,
                -ca*sg - cb*cg*sa, -sa*sg + ca*cb*cg,  cg*sb,
                 sa*sb,            -ca*sb,             cb
            });
    }

    [[nodiscard]] static constexpr RotationMatrix
    product_axes_zyz(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        const T ca = std::cos(alpha);
        const T sa = std::sin(alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos(gamma);
        const T sg = std::sin(gamma);
        if constexpr (layout == MatrixLayout::row_major && action == Action::passive)
            return RotationMatrix({
                -sa*sg + ca*cb*cg,  cg*sa + ca*cb*sg, -ca*sb,
                -ca*sg - cb*cg*sa,  ca*cg - cb*sa*sg,  sa*sb,
                 cg*sb,             sb*sg,             cb
            });
        else if constexpr (layout == MatrixLayout::column_major && action == Action::passive)
            return RotationMatrix({
                -sa*sg + ca*cb*cg, -ca*sg - cb*cg*sa,  cg*sb,
                 cg*sa + ca*cb*sg,  ca*cg - cb*sa*sg,  sb*sg,
                -ca*sb,             sa*sb,             cb
            });
        if constexpr (layout == MatrixLayout::row_major && action == Action::active)
            return RotationMatrix({
                -sa*sg + ca*cb*cg, -cg*sa - ca*cb*sg,  ca*sb,
                 ca*sg + cb*cg*sa,  ca*cg - cb*sa*sg,  sa*sb,
                -cg*sb,             sb*sg,             cb
            });
        else if constexpr (layout == MatrixLayout::column_major && action == Action::active)
            return RotationMatrix({
                -sa*sg + ca*cb*cg,  ca*sg + cb*cg*sa, -cg*sb,
                -cg*sa - ca*cb*sg,  ca*cg - cb*sa*sg,  sb*sg,
                 ca*sb,             sa*sb,             cb
            });
    }

    template <Order order>
    [[nodiscard]] static constexpr RotationMatrix
    product_axes_xyz(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        const T ca = std::cos((order == Order::reverse) ? gamma : alpha);
        const T sa = std::sin((order == Order::reverse) ? gamma : alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos((order == Order::reverse) ? alpha : gamma);
        const T sg = std::sin((order == Order::reverse) ? alpha : gamma);
        if constexpr (
                (layout == MatrixLayout::row_major && action == Action::passive && order == Order::keep)
                || (layout == MatrixLayout::column_major && action == Action::active && order == Order::reverse))
            return RotationMatrix({
                 cb*cg,             cb*sg,            -sb,
                -ca*sg + cg*sa*sb,  ca*cg + sa*sb*sg,  cb*sa,
                 sa*sg + ca*cg*sb, -cg*sa + ca*sb*sg,  ca*cb
            });
        else if constexpr (
                (layout == MatrixLayout::column_major && action == Action::passive && order == Order::keep)
                || (layout == MatrixLayout::row_major && action == Action::active && order == Order::reverse))
            return RotationMatrix({
                 cb*cg,            -ca*sg + cg*sa*sb,  sa*sg + ca*cg*sb,
                 cb*sg,             ca*cg + sa*sb*sg, -cg*sa + ca*sb*sg,
                -sb,                cb*sa,             ca*cb
            });
        if constexpr (
                (layout == MatrixLayout::row_major && action == Action::active && order == Order::keep)
                || (layout == MatrixLayout::column_major && action == Action::passive && order == Order::reverse))
            return RotationMatrix({
                 cb*cg,            -cb*sg,             sb,
                 ca*sg + cg*sa*sb,  ca*cg - sa*sb*sg, -cb*sa,
                 sa*sg - ca*cg*sb,  cg*sa + ca*sb*sg,  ca*cb
            });
        else if constexpr (
                (layout == MatrixLayout::column_major && action == Action::active && order == Order::keep)
                || (layout == MatrixLayout::row_major && action == Action::passive && order == Order::reverse))
            return RotationMatrix({
                 cb*cg,             ca*sg + cg*sa*sb,  sa*sg - ca*cg*sb,
                -cb*sg,             ca*cg - sa*sb*sg,  cg*sa + ca*sb*sg,
                 sb,               -cb*sa,             ca*cb
            });
    }

    template <Order order>
    [[nodiscard]] static constexpr RotationMatrix
    product_axes_xzy(T alpha, T beta, T gamma) noexcept requires(N == 3)
    {
        const T ca = std::cos((order == Order::reverse) ? gamma : alpha);
        const T sa = std::sin((order == Order::reverse) ? gamma : alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos((order == Order::reverse) ? alpha : gamma);
        const T sg = std::sin((order == Order::reverse) ? alpha : gamma);
        if constexpr (
                (layout == MatrixLayout::row_major && action == Action::passive && order == Order::keep)
                || (layout == MatrixLayout::column_major && action == Action::active && order == Order::reverse))
            return RotationMatrix({
                 cb*cg,             sb,               -cb*sg,
                 sa*sg - ca*cg*sb,  ca*cb,             cg*sa + ca*sb*sg,
                 ca*sg + cg*sa*sb, -cb*sa,             ca*cg - sa*sb*sg
            });
        else if constexpr (
                (layout == MatrixLayout::column_major && action == Action::passive && order == Order::keep)
                || (layout == MatrixLayout::row_major && action == Action::active && order == Order::reverse))
            return RotationMatrix({
                 cb*cg,             sa*sg - ca*cg*sb,  ca*sg + cg*sa*sb,
                 sb,                ca*cb,            -cb*sa,
                -cb*sg,             cg*sa + ca*sb*sg,  ca*cg - sa*sb*sg
            });
        if constexpr (
                (layout == MatrixLayout::row_major && action == Action::active && order == Order::keep)
                || (layout == MatrixLayout::column_major && action == Action::passive && order == Order::reverse))
            return RotationMatrix({
                 cb*cg,            -sb,                cb*sg,
                 sa*sg + ca*cg*sb,  ca*cb,            -cg*sa + ca*sb*sg,
                -ca*sg + cg*sa*sb,  cb*sa,             ca*cg + sa*sb*sg
            });
        else if constexpr (
                (layout == MatrixLayout::column_major && action == Action::active && order == Order::keep)
                || (layout == MatrixLayout::row_major && action == Action::passive && order == Order::reverse))
            return RotationMatrix({
                 cb*cg,             sa*sg + ca*cg*sb, -ca*sg + cg*sa*sb,
                -sb,                ca*cb,             cb*sa,
                 cb*sg,            -cg*sa + ca*sb*sg,  ca*cg + sa*sb*sg
            });
    }

    template <Order order>
    [[nodiscard]] static constexpr RotationMatrix
    product_axes_yxz(T alpha, T beta, T gamma) noexcept requires (N == 3)
    {
        const T ca = std::cos((order == Order::reverse) ? gamma : alpha);
        const T sa = std::sin((order == Order::reverse) ? gamma : alpha);
        const T cb = std::cos(beta);
        const T sb = std::sin(beta);
        const T cg = std::cos((order == Order::reverse) ? alpha : gamma);
        const T sg = std::sin((order == Order::reverse) ? alpha : gamma);
        if constexpr (
                (layout == MatrixLayout::row_major && action == Action::passive && order == Order::keep)
                || (layout == MatrixLayout::column_major && action == Action::active && order == Order::reverse))
            return RotationMatrix({
                 ca*cg - sa*sb*sg,  ca*sg + cg*sa*sb, -cb*sa,
                -cb*sg,             cb*cg,             sb,
                 cg*sa + ca*sb*sg,  sa*sg - ca*cg*sb,  ca*cb
            });
        else if constexpr (
                (layout == MatrixLayout::column_major && action == Action::passive && order == Order::keep)
                || (layout == MatrixLayout::row_major && action == Action::active && order == Order::reverse))
            return RotationMatrix({
                 ca*cg - sa*sb*sg, -cb*sg,             cg*sa + ca*sb*sg,
                 ca*sg + cg*sa*sb,  cb*cg,             sa*sg - ca*cg*sb,
                -cb*sa,             sb,                ca*cb
            });
        if constexpr (
                (layout == MatrixLayout::row_major && action == Action::active && order == Order::keep)
                || (layout == MatrixLayout::column_major && action == Action::passive && order == Order::reverse))
            return RotationMatrix({
                 ca*cg + sa*sb*sg, -ca*sg + cg*sa*sb,  cb*sa,
                 cb*sg,             cb*cg,            -sb,
                -cg*sa + ca*sb*sg,  sa*sg + ca*cg*sb,  ca*cb
            });
        else if constexpr (
                (layout == MatrixLayout::column_major && action == Action::active && order == Order::keep)
                || (layout == MatrixLayout::row_major && action == Action::passive && order == Order::reverse))
            return RotationMatrix({
                 ca*cg + sa*sb*sg,  cb*sg,            -cg*sa + ca*sb*sg,
                -ca*sg + cg*sa*sb,  cb*cg,             sa*sg + ca*cg*sb,
                 cb*sa,            -sb,                ca*cb
            });
    }

    Matrix<T, N, N, action, layout> m_matrix;
};

template <
    std::floating_point T, std::size_t N,
    Action action_param = Action::passive,
    MatrixLayout layout_param = MatrixLayout::column_major>
class RigidMatrix
{
public:
    using value_type = T;
    using index_type = std::size_t;
    using size_type = std::size_t;
    using transpose_type = Matrix<T, N + 1, N + 1, action_param, layout_param>;

    static constexpr Action action = action_param;
    static constexpr MatrixLayout layout = layout_param;
    static constexpr std::array<size_type, 2> shape = {N + 1, N + 1};

    constexpr RigidMatrix() = default;
    constexpr explicit RigidMatrix(std::array<T, (N + 1)*(N + 1)> array): m_matrix{array} {}

    [[nodiscard]] static constexpr RigidMatrix
    identity() noexcept
    {
        RigidMatrix res{};
        for (std::size_t i = 0; i < N + 1; ++i)
            res[i, i] = 1.0;
    }

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
    Matrix<T, N + 1, N + 1, action, layout> m_matrix;
};

template <
    std::floating_point T, std::size_t N,
    Action action_param = Action::passive,
    MatrixLayout matrix_layout_param = MatrixLayout::column_major
>
class RigidTransform
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using vector_type = Vector<T, N>;
    using rotation_matrix_type = RotationMatrix<T, N, action_param, matrix_layout_param>;

    static constexpr Action action = action_param;
    static constexpr MatrixLayout matrix_layout = matrix_layout_param;

    constexpr RigidTransform() = default;
    explicit constexpr RigidTransform(rotation_matrix_type rotation, vector_type translation):
        m_rotation{rotation}, m_translation{translation} {}

    [[nodiscard]] static constexpr RigidTransform
    identity() noexcept
    {
        return RigidTransform(rotation_matrix_type::identity(), vector_type{});
    }

    [[nodiscard]] constexpr vector_type operator()(const vector_type& vector) const noexcept
    {
        return matmul(m_rotation, vector) + m_translation;
    }

    template <matrix_like M>
    [[nodiscard]] constexpr auto operator()(const M& matrix) const noexcept
    {
        return matmul(m_rotation, matmul(matrix, transpose(m_rotation)));
    }

    [[nodiscard]] constexpr const rotation_matrix_type&
    rotation() const noexcept { return m_rotation; }

    [[nodiscard]] constexpr const vector_type&
    translation() const noexcept { return m_translation; }

    [[nodiscard]] constexpr RigidTransform
    inverse() const noexcept
    {
        return RigidTransform(m_rotation.inverse(), -(m_rotation.inverse()*m_translation));
    }

private:
    rotation_matrix_type m_rotation;
    vector_type m_translation;
};

template <
    std::floating_point T, std::size_t N,
    Action action = Action::passive,
    MatrixLayout matrix_layout = MatrixLayout::column_major
>
[[nodiscard]] constexpr auto
compose(
    const RigidTransform<T, N, action, matrix_layout>& a,
    const RigidTransform<T, N, action, matrix_layout>& b) noexcept
{
    return RigidTransform<T, N, action, matrix_layout>(
        b.rotation()*a.rotation(),
        b.rotation()*a.translation() + b.translation());
}

template <
    std::floating_point T, std::size_t N,
    Action action = Action::passive,
    MatrixLayout matrix_layout = MatrixLayout::column_major
>
[[nodiscard]] constexpr auto
compose(
    const RotationMatrix<T, N, action, matrix_layout>& rotation,
    const RigidTransform<T, N, action, matrix_layout>& rigid_transform) noexcept
{
    return RigidTransform<T, N, action, matrix_layout>(
        rigid_transform.rotation()*rotation, rigid_transform.translation());
}

template <
    std::floating_point T, std::size_t N,
    Action action = Action::passive,
    MatrixLayout matrix_layout = MatrixLayout::column_major
>
[[nodiscard]] constexpr auto
compose(
    const RigidTransform<T, N, action, matrix_layout>& rigid_transform,
    const RotationMatrix<T, N, action, matrix_layout>& rotation) noexcept
{
    return RigidTransform<T, N, action, matrix_layout>(
        rotation*rigid_transform.rotation(), rotation*rigid_transform.translation());
}

template <
    std::floating_point T, std::size_t N,
    Action action = Action::passive,
    MatrixLayout matrix_layout = MatrixLayout::column_major
>
[[nodiscard]] constexpr auto
compose(
    const Vector<T, N>& translation,
    const RigidTransform<T, N, action, matrix_layout>& rigid_transform)
{
    return RigidTransform<T, N, action, matrix_layout>(
        rigid_transform.rotation(), rigid_transform.translation() + rigid_transform.rotation()*translation);
}

template <
    std::floating_point T, std::size_t N,
    Action action = Action::passive,
    MatrixLayout matrix_layout = MatrixLayout::column_major
>
[[nodiscard]] constexpr auto
compose(
    const RigidTransform<T, N, action, matrix_layout>& rigid_transform,
    const Vector<T, N>& translation)
{
    return RigidTransform<T, N, action, matrix_layout>(
        rigid_transform.rotation(), rigid_transform.translation() + translation);
}

template <
    std::floating_point T, std::size_t N,
    Action action = Action::passive,
    MatrixLayout matrix_layout = MatrixLayout::column_major
>
[[nodiscard]] constexpr auto
compose(
    const RotationMatrix<T, N, action, matrix_layout>& rotation,
    const Vector<T, N>& translation)
{
    return RigidTransform<T, N, action, matrix_layout>(rotation, translation);
}

template <
    std::floating_point T, std::size_t N,
    Action action = Action::passive,
    MatrixLayout matrix_layout = MatrixLayout::column_major
>
[[nodiscard]] constexpr auto
compose(
    const Vector<T, N>& translation,
    const RotationMatrix<T, N, action, matrix_layout>& rotation)
{
    return RigidTransform<T, N, action, matrix_layout>(rotation, rotation*translation);
}

} // namespace la

} // namespace zdm
