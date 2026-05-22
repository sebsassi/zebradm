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

#include <cstddef>

#include "identity.hpp"
#include "rotation.hpp"
#include "transform_conventions.hpp"
#include "translation.hpp"

namespace zdm::la
{

/**
    @brief A type representing a rigid transform.

    @tparam T Value type of the transform.
    @tparam N Dimension of the transform.
    @tparam action_param action convention.
    @tparam matrix_layout_param Matrix layout convention.

    This type represents a rigid transform, which is a combination of a rotation
    and a translation. Specifically, given a vector \f$\vec{v}\f$ and a rigid
    transform with rotation \f$R\f$ and translation \f$\vec{u}\f$, the
    transformed vector is given by \f$\vec{v}' = R\vec{v} + \vec{u}'.
*/
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
    using translation_type = Translation<T, N, action_param>;
    using rotation_matrix_type = RotationMatrix<T, N, action_param, matrix_layout_param>;
    using vector_type = typename translation_type::vector_type;

    static constexpr Action action = action_param;
    static constexpr MatrixLayout matrix_layout = matrix_layout_param;

    constexpr RigidTransform() = default;

    explicit constexpr RigidTransform([[maybe_unused]] Identity id):
        RigidTransform(rotation_matrix_type::identity(), translation_type::identity()) {}

    explicit constexpr RigidTransform(const rotation_matrix_type& matrix):
        RigidTransform(matrix, translation_type::identity()) {}

    explicit constexpr RigidTransform(const translation_type& translation):
        RigidTransform(rotation_matrix_type::identity(), translation) {}

    explicit constexpr RigidTransform(const vector_type& vector):
        RigidTransform(rotation_matrix_type::identity(), translation_type{vector}) {}

    constexpr RigidTransform(const rotation_matrix_type& rotation, const translation_type& translation):
        m_rotation{rotation}, m_translation{translation} {}

    constexpr RigidTransform(const rotation_matrix_type& rotation, const vector_type& translation):
        m_rotation{rotation}, m_translation{translation_type{translation}} {}

    /**
        @brief Create an identity rigid transform.

        @return The identity rigid transform \f$(R, \vec{u}) = (I, 0)\f$.
    */
    [[nodiscard]] static constexpr RigidTransform
    identity() noexcept
    {
        return RigidTransform(rotation_matrix_type::identity(), translation_type::identity());
    }

    /**
        @brief Construct a rigid transform from a rotation followed by a
        translation.

        @tparam chaining Transformation chaining conventioin (intrinsic vs.
        extrinsic).

        @param rotation Rotation matrix \f$R\f$.
        @param translation Translation vector \f$\vec{u}\f$.

        @return The composite rigid tranasform \f$(R \circ T_\vec{u})\f$.

        This method gives the rigid transform corresponding to the composition
        \f$(R \circ T_\vec{u})\f$ of the rotation $R$ and a translation
        $T_\vec{u}$ (where $T_\vec{u}(\vec{v}) = \vec{v} + \vec{u}$). How this
        composition operates on the components of a fector depends on the
        chosen composition convention (intrinsic vs. extrinsic), as well as on
        the action convention (active vs. passive). Namely, the corresponding
        transforms operating on a vector \f$\vec{v}\f$ are given by the
        following table

                | intrinsic                 | extrinsic
        active  | \f$R\vec{v} + R\vec{u}\f$ | \f$R\vec{v} + \vec{u}\f$
        passive | \f$R\vec{v} - \vec{u}\f$  | \f$R\vec{v} - R\vec{u}\f$
    */
    template <Chaining chaining>
    [[nodiscard]] static constexpr RigidTransform
    from(const rotation_matrix_type& rotation, const translation_type& translation)
    {
        if constexpr (
                (chaining == Chaining::extrinsic && action == Action::active)
                || (chaining == Chaining::intrinsic && action == Action::passive))
            return RigidTransform{rotation, translation};
        else
            return RigidTransform{rotation, rotation*translation};
    }

    template <Chaining chaining>
    [[nodiscard]] static constexpr RigidTransform
    from(const rotation_matrix_type& rotation, const vector_type& translation)
    {
        return from<chaining>(rotation, translation_type{translation});
    }

    /**
        @brief Construct a rigid transform from a translation followed by a
        rotation.

        @tparam chaining Transformation chaining conventioin (intrinsic vs.
        extrinsic).

        @param translation Translation vector \f$\vec{u}\f$.
        @param rotation Rotation matrix \f$R\f$.

        @return The composite rigid tranasform \f$(T_\vec{u} \circ R)\f$.

        This method gives the rigid transform corresponding to the composition
        \f$(T_\vec{u} \circ R)\f$ of the rotation $R$ and a translation
        $T_\vec{u}$ (where $T_\vec{u}(\vec{v}) = \vec{v} + \vec{u}$). How this
        composition operates on the components of a fector depends on the
        chosen composition convention (intrinsic vs. extrinsic), as well as on
        the action convention (active vs. passive). Namely, the corresponding
        transforms operating on a vector \f$\vec{v}\f$ are given by the
        following table

                | intrinsic                 | extrinsic
        active  | \f$R\vec{v} + \vec{u}\f$  | \f$R\vec{v} + R\vec{u}\f$
        passive | \f$R\vec{v} - R\vec{u}\f$ | \f$R\vec{v} - \vec{u}\f$
    */
    template <Chaining chaining>
    [[nodiscard]] static constexpr RigidTransform
    from(const translation_type& translation, const rotation_matrix_type& rotation)
    {
        if constexpr (
                (chaining == Chaining::extrinsic && action == Action::active)
                || (chaining == Chaining::intrinsic && action == Action::passive))
            return RigidTransform{rotation, rotation*translation};
        else
            return RigidTransform{rotation, translation};
    }

    template <Chaining chaining>
    [[nodiscard]] static constexpr RigidTransform
    from(const vector_type& translation, const rotation_matrix_type& rotation)
    {
        return from<chaining>(translation_type{translation}, rotation);
    }

    /**
        @brief Construct a rigid transform from a rotation.

        @param rotation

        @return Rigid transformation corresponding to the rotation.
    */
    [[nodiscard]] static constexpr RigidTransform
    from(const rotation_matrix_type& rotation)
    {
        return RigidTransform{rotation};
    }

    /**
        @brief Construct a rigid transform from a translation.

        @param rotation

        @return Rigid transform corresponding to the translation.
    */
    [[nodiscard]] static constexpr RigidTransform
    from(const translation_type& translation)
    {
        return RigidTransform{translation};
    }

    [[nodiscard]] static constexpr RigidTransform
    from(const vector_type& translation)
    {
        return from(translation_type{translation});
    }

    [[nodiscard]] constexpr bool operator==(const RigidTransform& other) const noexcept = default;

    /**
        @brief Transform a vector with the rigid transform.

        @param vector Vector to be transformed.

        @return Transformed vector.
    */
    [[nodiscard]] constexpr vector_type operator()(const vector_type& vector) const noexcept
    {
        return m_translation(matmul(m_rotation, vector));
    }

    /**
        @brief Rotate a matrix with the rigid transform.

        @param matrix Matrix to be rotated.

        @return Rotated matrix.
    */
    template <static_matrix_like M>
    [[nodiscard]] constexpr auto operator()(const M& matrix) const noexcept
    {
        return matmul(m_rotation, matmul(matrix, transpose(m_rotation)));
    }

    /**
        @brief Rotation part of the transform.

        @return Rotation matrix holding the rotation part of the transform.
    */
    [[nodiscard]] constexpr const rotation_matrix_type&
    rotation() const noexcept { return m_rotation; }

    /**
        @brief Translation part of the transform.

        @return Vector holding the translation part of the transform.
    */
    [[nodiscard]] constexpr const translation_type&
    translation() const noexcept { return m_translation; }

    /**
        @brief Inverse of the transform.

        @return Rigid transform corresponding to the inverse of the transform.
    */
    [[nodiscard]] constexpr RigidTransform
    inverse() const noexcept
    {
        return RigidTransform(m_rotation.inverse(), -(m_rotation.inverse()*m_translation));
    }

private:

    rotation_matrix_type m_rotation;
    translation_type m_translation;
};

/**
    @brief Compose two rigid transforms.

    @tparam chaining Transformation chaining conventioin (intrinsic vs.
    extrinsic).
    @tparam T Value type of the transforms.
    @tparam N Dimension of the transforms.
    @tparam action_param action convention.
    @tparam matrix_layout_param Matrix layout convention.

    @param a First rigid transform.
    @param b Second rigid transform.

    @return Composite rigid transform.
*/
template <
    Chaining chaining,
    std::floating_point T, std::size_t N,
    Action action,
    MatrixLayout matrix_layout
>
[[nodiscard]] constexpr RigidTransform<T, N, action, matrix_layout>
compose(
    const RigidTransform<T, N, action, matrix_layout>& a,
    const RigidTransform<T, N, action, matrix_layout>& b) noexcept
{
    if constexpr (
            (chaining == Chaining::extrinsic && action == Action::active)
            || (chaining == Chaining::intrinsic && action == Action::passive))
        return RigidTransform<T, N, action, matrix_layout>::template from<chaining>(
            b.rotation()*a.rotation(),
            b.rotation()*a.translation() + b.translation());
    else
        return RigidTransform<T, N, action, matrix_layout>::template from<chaining>(
            a.rotation()*b.rotation(),
            a.rotation()*b.translation() + a.translation());
}

/**
    @brief Compose a rotation with a rigid transform.

    @tparam chaining Transformation chaining conventioin (intrinsic vs.
    extrinsic).
    @tparam T Value type of the transforms.
    @tparam N Dimension of the transforms.
    @tparam action_param action convention.
    @tparam matrix_layout_param Matrix layout convention.

    @param rotation
    @param rigid_transform

    @return Composite rigid transform.
*/
template <
    Chaining chaining,
    std::floating_point T, std::size_t N,
    Action action,
    MatrixLayout matrix_layout
>
[[nodiscard]] constexpr auto
compose(
    const RotationMatrix<T, N, action, matrix_layout>& rotation,
    const RigidTransform<T, N, action, matrix_layout>& rigid_transform) noexcept
{
    if constexpr (
            (chaining == Chaining::extrinsic && action == Action::active)
            || (chaining == Chaining::intrinsic && action == Action::passive))
        return RigidTransform<T, N, action, matrix_layout>::template from<Chaining::intrinsic>(
            rigid_transform.rotation()*rotation,
            rigid_transform.translation());
    else
        return RigidTransform<T, N, action, matrix_layout>::template from<Chaining::intrinsic>(
            rotation*rigid_transform.rotation(),
            rotation*rigid_transform.translation());
}

/**
    @brief Compose a rigid transform with a rotation.

    @tparam chaining Transformation chaining conventioin (intrinsic vs.
    extrinsic).
    @tparam T Value type of the transforms.
    @tparam N Dimension of the transforms.
    @tparam action_param action convention.
    @tparam matrix_layout_param Matrix layout convention.

    @param rigid_transform
    @param rotation

    @return Composite rigid transform.
*/
template <
    Chaining chaining,
    std::floating_point T, std::size_t N,
    Action action,
    MatrixLayout matrix_layout
>
[[nodiscard]] constexpr auto
compose(
    const RigidTransform<T, N, action, matrix_layout>& rigid_transform,
    const RotationMatrix<T, N, action, matrix_layout>& rotation) noexcept
{
    if constexpr (
            (chaining == Chaining::extrinsic && action == Action::active)
            || (chaining == Chaining::intrinsic && action == Action::passive))
        return RigidTransform<T, N, action, matrix_layout>::template from<Chaining::intrinsic>(
            rotation*rigid_transform.rotation(),
            rotation*rigid_transform.translation());
    else
        return RigidTransform<T, N, action, matrix_layout>::template from<Chaining::intrinsic>(
            rigid_transform.rotation()*rotation,
            rigid_transform.translation());
}

/**
    @brief Compose a translation with a rigid transform.

    @tparam chaining Transformation chaining conventioin (intrinsic vs.
    extrinsic).
    @tparam T Value type of the transforms.
    @tparam N Dimension of the transforms.
    @tparam action_param action convention.
    @tparam matrix_layout_param Matrix layout convention.

    @param translation
    @param rigid_transform

    @return Composite rigid transform.
*/
template <
    Chaining chaining,
    std::floating_point T, std::size_t N,
    Action action,
    MatrixLayout matrix_layout
>
[[nodiscard]] constexpr auto
compose(
    const Translation<T, N, action>& translation,
    const RigidTransform<T, N, action, matrix_layout>& rigid_transform)
{
    if constexpr (
            (chaining == Chaining::extrinsic && action == Action::active)
            || (chaining == Chaining::intrinsic && action == Action::passive))
        return RigidTransform<T, N, action, matrix_layout>::template from<Chaining::intrinsic>(
            rigid_transform.rotation(),
            rigid_transform.rotation()*translation + rigid_transform.translation());
    else
        return RigidTransform<T, N, action, matrix_layout>::template from<Chaining::extrinsic>(
            rigid_transform.rotation(),
            rigid_transform.translation() + translation);
}

/**
    @brief Compose a rigid transform with a translation.

    @tparam chaining Transformation chaining conventioin (intrinsic vs.
    extrinsic).
    @tparam T Value type of the transforms.
    @tparam N Dimension of the transforms.
    @tparam action_param action convention.
    @tparam matrix_layout_param Matrix layout convention.

    @param rigid_transform
    @param translation

    @return Composite rigid transform.
*/
template <
    Chaining chaining,
    std::floating_point T, std::size_t N,
    Action action,
    MatrixLayout matrix_layout
>
[[nodiscard]] constexpr auto
compose(
    const RigidTransform<T, N, action, matrix_layout>& rigid_transform,
    const Translation<T, N, action>& translation)
{
    if constexpr (
            (chaining == Chaining::extrinsic && action == Action::active)
            || (chaining == Chaining::intrinsic && action == Action::passive))
        return RigidTransform<T, N, action, matrix_layout>::template from<Chaining::extrinsic>(
            rigid_transform.rotation(),
            rigid_transform.translation() + translation);
    else
        return RigidTransform<T, N, action, matrix_layout>::template from<Chaining::intrinsic>(
            rigid_transform.rotation(),
            rigid_transform.rotation()*translation + rigid_transform.translation());
}

/**
    @brief Compose a rotation with a translation into a rigid transform.

    @tparam chaining Transformation chaining conventioin (intrinsic vs.
    extrinsic).
    @tparam T Value type of the transforms.
    @tparam N Dimension of the transforms.
    @tparam action_param action convention.
    @tparam matrix_layout_param Matrix layout convention.

    @param rotation
    @param translation

    @return Composite rigid transform.
*/
template <
    Chaining chaining,
    std::floating_point T, std::size_t N,
    Action action,
    MatrixLayout matrix_layout
>
[[nodiscard]] constexpr auto
compose(
    const RotationMatrix<T, N, action, matrix_layout>& rotation,
    const Translation<T, N, action>& translation)
{
    return RigidTransform<T, N, action, matrix_layout>::template from<chaining>(rotation, translation);
}

/**
    @brief Compose a translation with a rotation into a rigid transform.

    @tparam chaining Transformation chaining conventioin (intrinsic vs.
    extrinsic).
    @tparam T Value type of the transforms.
    @tparam N Dimension of the transforms.
    @tparam action_param action convention.
    @tparam matrix_layout_param Matrix layout convention.

    @param translation
    @param rotation

    @return Composite rigid transform.
*/
template <
    Chaining chaining,
    std::floating_point T, std::size_t N,
    Action action,
    MatrixLayout matrix_layout
>
[[nodiscard]] constexpr auto
compose(
    const Translation<T, N, action>& translation,
    const RotationMatrix<T, N, action, matrix_layout>& rotation)
{
    return RigidTransform<T, N, action, matrix_layout>::template from<chaining>(translation, rotation);
}

} // namspace zdm::la
