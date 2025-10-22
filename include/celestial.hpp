#pragma once

#include <tuple>

#include "matrix.hpp"
#include "astro.hpp"

namespace zdm
{

template <typename ParametricTransform, typename OutputType>
concept outputs = requires (ParametricTransform x, typename ParametricTransform::value_type t)
    {
        { x(t) } -> std::same_as<OutputType>;
    };

template <typename T>
concept parametric_rigid_transform
    = outputs<T, la::RigidTransform<typename T::value_type, T::dimension, T::action, T::matrix_layout>>;

template <typename RigidTransformType, parametric_rigid_transform... Types>
    requires (outputs<Types, RigidTransformType> && ...)
class CompositeRigidTransform
{
public:
    using rigid_transform_type = RigidTransformType;

    CompositeRigidTransform(const Types&... transforms): m_transforms(transforms...) {}

    [[nodiscard]] rigid_transform_type operator()(const rigid_transform_type::value_type& parameter) const noexcept
    {
        return std::apply([&](Types... transforms)
        {
            auto res = rigid_transform_type::identity();
            ([&]{ res = la::compose(res, transforms(parameter)); }(), ...);
            return res;
        }, m_transforms);
    }

private:
    std::tuple<Types...> m_transforms;
};

class GCSToICRS
{
public:
    constexpr GCSToICRS(
        double circular_velocity,
        const la::Vector<double, 3>& peculiar_velocity = peculiar_velocity_sbd_2010,
        GalacticOrientation orientation = orientation_km_2017)
    {
        const la::Vector<double, 3> boost = peculiar_velocity + la::Vector<double, 3>{0.0, circular_velocity, 0.0};
        m_transform = la::RigidTransform<double, 3>(orientation.gcs_to_icrs(), boost);
    };

    la::RigidTransform<double, 3> operator()([[maybe_unused]] double t) { return m_transform; }
    la::RigidTransform<double, 3> operator()() { return m_transform; }

private:
    la::RigidTransform<double, 3> m_transform;
};

class ECSToICRS
{
public:
    constexpr ECSToICRS() = default;

    la::RigidTransform<double, 3> operator()([[maybe_unused]] double t) { return transform(); }
    la::RigidTransform<double, 3> operator()() { return transform(); }

private:
    [[nodiscard]] static constexpr la::RigidTransform<double, 3>
    transform() noexcept
    {

    }
};

} // namespace zdm
