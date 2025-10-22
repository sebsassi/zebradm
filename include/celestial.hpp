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

template <std::floating_point T, std::size_t N, parametric_rigid_transform... Types>
    requires (outputs<Types, la::RigidTransform<T, N>> && ...)
class CompositeRigidTransform
{
public:
    using rigid_transform_type = la::RigidTransform<T, N>;

    CompositeRigidTransform(const Types&... transforms): m_transforms(transforms...) {}

    [[nodiscard]] rigid_transform_type
    operator()(const rigid_transform_type::value_type& parameter) const noexcept
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
        GalacticOrientation orientation = orientation_km_2017):
        m_transform(
            orientation.gcs_to_icrs(),
            peculiar_velocity + la::Vector<double, 3>{0.0, circular_velocity, 0.0}) {};

    la::RigidTransform<double, 3> operator()([[maybe_unused]] double t) { return m_transform; }
    la::RigidTransform<double, 3> operator()() { return m_transform; }

private:
    la::RigidTransform<double, 3> m_transform;
};

class ECSToICRS
{
public:
    constexpr ECSToICRS() = default;

    [[nodiscard]] constexpr la::RigidTransform<double, 3>
    operator()([[maybe_unused]] double t) const noexcept { return transform(); }

    [[nodiscard]] constexpr la::RigidTransform<double, 3>
    operator()() const noexcept { return transform(); }

private:
    // This function can be consteval because the value of the J2000 obliquity
    // for the purposes of this program is fixed from the IAU 2009 System of
    // Astronomical Constants at 84381.406 arc seconds. This code can be
    // updated if a better estimate is ever adopted by the IAU. Not that it
    // matters, because this code really doesn't require mas level accuracy.
    [[nodiscard]] static constexpr la::RigidTransform<double, 3>
    transform() noexcept
    {
        // Cosine and sine of the J2000 obliquity of the ecliptic.
        constexpr double cos_obliquity_j2000 = 0.9174821430652418;
        constexpr double sin_obliquity_j2000 = 0.4090926006005829;
        constexpr auto res = la::RigidTransform<double, 3>(
            la::RotationMatrix<double, 3>({
                1.0,  0.0,                 0.0,
                0.0,  cos_obliquity_j2000, sin_obliquity_j2000,
                0.0, -sin_obliquity_j2000, cos_obliquity_j2000,
            }),
            la::Vector<double, 3>{});
        return res;
    }
};

class ICRSToGCRS
{
public:
    constexpr ICRSToGCRS() = default;

    [[nodiscard]] la::RigidTransform<double, 3>
    operator()(double t) const noexcept
    {
        const la::Vector<double, 3> earth_velocity = zdm::earth_orbit(t).reference_cs_velocity();
        return la::RigidTransform<double, 3>(
            la::RotationMatrix<double, 3>::identity(),
            ecs_to_icrs().rotation()*(-earth_velocity));
    }

private:
    static constexpr auto ecs_to_icrs = ECSToICRS{};
};

class GCRSToITRS;

class ITRSToHCS;

} // namespace zdm
