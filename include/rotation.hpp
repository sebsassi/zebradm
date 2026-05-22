/*
Copyright (c) 2025 Sebastian Sassi

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


#include <cmath>

#include "transform_conventions.hpp"
#include "matrix.hpp"

namespace zdm::la
{

/**
    @brief Enum specifying an Euler angle convention.
*/
enum class EulerConvention
{
    xzx,
    xyx,
    yxy,
    yzy,
    zyz,
    zxz
};

/**
    @brief Enum specifying a Tait-Bryan angle convention.
*/
enum class TaitBryanConvention
{
    xyz,
    xzy,
    yxz,
    yzx,
    zxy,
    zyx
};

/**
    @brief A type representing a rotation matirx.

    @tparam T Type of matrix elements.
    @tparam N Dimension of matrix.
    @tparam action_param Matrix action convention.
    @tparam layout_param Matrix layout convention.

    A matrix \f$M\f$ is a rotation matrix if it is orthogonal, i.e.,
    \f$M^\mathsf{T}M = MM^\mathsf{T} = I\f$, and \f$\det M = 1\f$.
*/
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
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using transpose_type = RotationMatrix;

    static constexpr Action action = action_param;
    static constexpr MatrixLayout layout = layout_param;
    static constexpr std::array<size_type, 2> shape = {N, N};

    constexpr explicit RotationMatrix() = default;
    constexpr explicit RotationMatrix(std::array<T, N*N> array): m_matrix{array} {}

    /**
        @brief Create an identity matrix.
    */
    [[nodiscard]] static constexpr RotationMatrix
    identity() noexcept
    {
        RotationMatrix res{};
        for (std::size_t i = 0; i < N; ++i)
            res[i, i] = 1.0;
        return res;
    }

    /**
        @brief Create a \f$2\times2\f$ rotation matrix from a rotaton angle.

        @param angle Rotation angle \f$\theta\f$.

        @return \f$2\times 2\f$ rotation matrix \f$R(\theta)\f$.

        The resulting rotation matrix \f$R(\theta)\f$ depends on the chosen
        action convention of the matrix. For active rotations we have
        \f[
            R(\theta) =
            \begin{matrix}
                \cos\theta & -\sin\theta \\
                \sin\theta & \cos\theta
            \end{pmatrix}
        \f]
        For passive rotations
        \f[
            R(\theta) =
            \begin{matrix}
                \cos\theta & \sin\theta \\
                -\sin\theta & \cos\theta
            \end{pmatrix}
        \f]

    */
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

    /**
        @brief Create a \f$3\times3\f$ rotation matrix from a rotaton angle
        around a coordinate axis.

        @tparam axis Coordinate axis.

        @param angle Rotation angle.

        @return \f$3\times 3\f$ rotation matrix \f$R_i(\theta)\f$, where \f$i =
        x, y, z\f$.

        The resulting rotation matrix depends on the chosen action convention.
        For active rotations we have
        \f[
            R_x(\theta) =
            \begin{pmatrix}
                1 & 0          & 0           \\
                0 & \cos\theta & -\sin\theta \\
                0 & \sin\theta & \cos\theta
            \end{pmatrix},\quad
            R_y(\theta) =
            \begin{pmatrix}
                \cos\theta  & 0 & \sin\theta \\
                0           & 1 & 0          \\
                -\sin\theta & 0 & \cos\theta
            \end{pmatrix},\quad
            R_z(\theta) =
            \begin{pmatrix}
                \cos\theta & -\sin\theta & 0 \\
                \sin\theta & \cos\theta  & 0 \\
                0          & 0           & 1
            \end{pmatrix}.
        \f]
        For poassive rotations
        \f[
            R_x(\theta) =
            \begin{pmatrix}
                1 & 0           & 0           \\
                0 & \cos\theta  & \sin\theta  \\
                0 & -\sin\theta & \cos\theta
            \end{pmatrix},\quad
            R_y(\theta) =
            \begin{pmatrix}
                \cos\theta & 0 & -\sin\theta \\
                0          & 1 & 0           \\
                \sin\theta & 0 & \cos\theta
            \end{pmatrix},\quad
            R_z(\theta) =
            \begin{pmatrix}
                \cos\theta  & \sin\theta & 0 \\
                -\sin\theta & \cos\theta & 0 \\
                0           & 0          & 1
            \end{pmatrix}.
        \f]
    */
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

    /**
        @brief Create a \f$3\times3\f$ rotation matrix from a rotaton axis and
        a rotation angle around that axis.

        @param axis Rotation axis \f$\hat{n}\f$.
        @param angle Rotation angle \f$\theta\f$.

        @return \f$3\times 3\f$ rotation matrix \f$R_{\hat{n}}(\theta)\f$.

        This method gives a rotation matrix presented in the axis-angle
        convention. If we denote by $K(\hat{n})$ the cross product matrix of
        the axis \f$\hat{n}\f$, that is, $K(\hat{n})\vec{v} =
        \hat{n}\times\vec{v}$, then the matrix representing an active rotation
        is given by
        \f[
            R_{\hat{n}} = I - K\sin\theta + K^2(1 - \cos\theta),
        \f]
        and the matrix representing a passive rotation is given by
        \f[
            R_{\hat{n}} = I + K\sin\theta + K^2(1 - \cos\theta).
        \f]
    */
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

    /**
        @brief Create a \f$3\times3\f$ rotation matrix corresponding to a
        product of two coordinate axis rotation matrices.

        @tparam axis_alpha Coordinate axis for the first angle.
        @tparam axis_beta Coordinate axis for the second angle.

        @param alpha First rotation angle.
        @param beta Second rotation angle.

        This function creates the rotation matrix corresponding to the product
        \f[
            R_i(\alpha)R_j(\beta),
        \f]
        where \f$i,j = X,Y,Z\f$ denotes coordinate axis.
    */
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

    /**
        @brief Create a \f$3\times3\f$ rotation matrix corresponding to a
        composition of two coordinate axis rotations.

        @tparam axis_alpha Coordinate axis for the first angle.
        @tparam axis_beta Coordinate axis for the second angle.
        @tparam chaining Transformation chaining conventioin (intrinsic vs.
        extrinsic).

        @param alpha First rotation angle.
        @param beta Second rotation angle.

        This function creates the rotation matrix corresponding to the
        composition of two rotations
        \f[
            R_i(\alpha)\circ R_j(\beta),
        \f]
        where \f$i,j = X,Y,Z\f$ denotes coordinate axis. Note that the order
        of matrix multiplication this composition corresponds to depends on
        the chosen composition convention (intrinsic vs. extrinsic), as well as
        on the action convention (active vs. passive) of the matrix. Namely,
        the multiplication orders are given by the following table

                | intrinsic                   | extrinsic
        active  | \f$R_i(\alpha)R_j(\beta)\f$ | \f$R_j(\beta)R_i(\alpha)\f$
        passive | \f$R_j(\beta)R_i(\alpha)\f$ | \f$R_i(\alpha)R_j(\beta)\f$
    */
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
        //      intrinsic, active: `R_i(alpha)R_j(beta)`
        //      intrinsic, passive: `R_j(beta)R_i(alpha)`
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

    /**
        @brief Create a \f$3\times3\f$ rotation matrix corresponding to a
        product of three coordinate axis rotation matrices.

        @tparam axis_alpha Coordinate axis for the first angle.
        @tparam axis_beta Coordinate axis for the second angle.
        @tparam axis_gamma Coordinate axis for the third angle.

        @param alpha First rotation angle.
        @param beta Second rotation angle.
        @param gamma Third rotation angle.

        This function creates the rotation matrix corresponding to the product
        \f[
            R_i(\alpha)R_j(\beta)R_k(\gamma),
        \f]
        where \f$i,j,k = X,Y,Z\f$ denotes a coordinate axis.
    */
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

    /**
        @brief Create a \f$3\times3\f$ rotation matrix from Euler angles.

        @tparam convention Convention for rotation axes of the Euler angles.
        @tparam chaining Transformation chaining conventioin (intrinsic vs.
        extrinsic).

        @param alpha First Euler angle.
        @param beta Second Euler angle.
        @param gamma Third Euler angle.

        This function creates a rotation matrix from a set of proper/classical
        Euler angles. All twelve conventions are supported via the template
        parameters. The angles are defined always in the order of rotation
        composition, that is, \f$\alpha\f$ is the angle of the first rotation,
        \f$\beta\f$ is the angle of the second, and \f$\gamma\f$ the angle of
        the final rotation.
    */
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

    /**
        @brief Create a \f$3\times3\f$ rotation matrix from Tait-Bryan angles.

        @tparam convention Convention for rotation axes of the Tait-Bryan
        angles.
        @tparam chaining Transformation chaining conventioin (intrinsic vs.
        extrinsic).

        @param alpha First Tait-Bryan angle.
        @param beta Second Tait-Bryan angle.
        @param gamma Third Tait-Bryan angle.

        This function creates a rotation matrix from a set of Tait-Bryan angles.
        All twelve conventions are supported via the template parameters. The
        angles are defined always in the order of rotation composition, that is,
        \f$\alpha\f$ is the angle of the first rotation, \f$\beta\f$ is the
        angle of the second, and \f$\gamma\f$ the angle of the final rotation.
    */
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

    /**
        @brief Create a \f$3\times3\f$ rotation matrix that aligns the z-axis
        with a given vector,

        @param vector corresponding to the new z-axis.

        This function creates a rotation matrix, which aligns the z-axis with
        the given vector. That is, given a vector \f$\vec{v}\f$ this function
        returns a rotation matrix \f$R_\vec{v}\f$such that \f$R_\vec{v}\vec{v}
        = \hat{z}\f$.
    */
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
            if constexpr (layout == MatrixLayout::column_major)
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
            if constexpr (layout == MatrixLayout::column_major)
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

    [[nodiscard]] constexpr operator Matrix<T, N, N, action, layout>() const noexcept { return m_matrix; }
    [[nodiscard]] constexpr explicit operator std::array<T, N*N>() const noexcept { return m_matrix.array; }

    [[nodiscard]] constexpr bool operator==(const RotationMatrix& other) const noexcept = default;

    [[nodiscard]] constexpr T&
    operator[](std::size_t i, std::size_t j) noexcept { return m_matrix[i, j]; }

    [[nodiscard]] constexpr const T&
    operator[](std::size_t i, std::size_t j) const noexcept { return m_matrix[i, j]; }

    template <static_vector_like V>
    [[nodiscard]] constexpr V
    operator()(const V& v) { return matmul(*this, v); }

    /**
        @brief Multiply two rotation matrices.
    */
    [[nodiscard]] constexpr RotationMatrix
    operator*(const RotationMatrix& other) const noexcept { return matmul(*this, other); }

    /**
        @brief Multiply rotation matrix by a general matrix.
    */
    [[nodiscard]] friend constexpr Matrix<T, N, N, action, layout>
    operator*(const RotationMatrix& r, const Matrix<T, N, N, action, layout>& m) { return matmul(r, m); }

    /**
        @brief Multiply rotation matrix by a general matrix.
    */
    [[nodiscard]] friend constexpr Matrix<T, N, N, action, layout>
    operator*(const Matrix<T, N, N, action, layout>& m, const RotationMatrix& r) { return matmul(m, r); }

    /**
        @brief Multiply a vector by a rotation matrix.
    */
    template <static_vector_like V>
    [[nodiscard]] constexpr V
    operator*(const V& vector) const noexcept { return matmul(*this, vector); }

    /**
        @brief Inverse of the transform.

        @return The transposed rotation matrix \f$R^\mathsf{T}\f$.
    */
    [[nodiscard]] constexpr RotationMatrix
    inverse() const noexcept
    {
        return transpose(*this);
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

/*
    @brief Composition of two rotations.

    @tparam chaining Transformation chaining conventioin (intrinsic vs.
    extrinsic).
    @tparam T Value type of the rotation matrix.
    @tparam N Dimension of the rotation matrix.
    @tparam action Matrix action convention.
    @tparam matrix_layout Matrix layout convention.

    @param a First rotation matrix \f$R_a\f$.
    @param b Second rotation matrix \f$R_b\f$.

    @return Composite rotation matrix \f$(R_a \circ R_b)\f$.

    This function gives the matrix corresponding to the composition
    \f$(R_a \circ R_b)\f$ of the two rotation matrices \f$R_a\f$ and \f$R_b\f$.
    The order of matrix multiplication this composition corresponds to depends
    on the chosen composition convention (intrinsic vs. extrinsic), as well as
    on the action convention (active vs. passive) of the matrices. Namely,
    the multiplication orders are given by the following table

            | intrinsic                   | extrinsic
    active  | \f$R_i(\alpha)R_j(\beta)\f$ | \f$R_j(\beta)R_i(\alpha)\f$
    passive | \f$R_j(\beta)R_i(\alpha)\f$ | \f$R_i(\alpha)R_j(\beta)\f$

*/
template <
    Chaining chaining,
    std::floating_point T, std::size_t N,
    Action action = Action::passive,
    MatrixLayout matrix_layout = MatrixLayout::column_major
>
[[nodiscard]] constexpr RotationMatrix<T, N, action, matrix_layout>
compose(
    const RotationMatrix<T, N, action, matrix_layout>& a,
    const RotationMatrix<T, N, action, matrix_layout>& b) noexcept
{
    if constexpr (
            (action == Action::passive && chaining == Chaining::intrinsic)
            || (action == Action::active && chaining == Chaining::extrinsic))
        return b*a;
    else
        return a*b;
}

} // namespace zdm::la
