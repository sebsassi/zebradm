#pragma once

#include <tuple>

#include "matrix.hpp"
#include "astro.hpp"

namespace zdm
{

namespace celestial
{

enum class CoordinateSystem
{
    GCS,
    ICRS,
    ITRS,
    HCS
};

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

class GCStoICRS
{
public:
    constexpr GCStoICRS(
        double circular_velocity,
        const la::Vector<double, 3>& peculiar_velocity = astro::peculiar_velocity_sbd_2010,
        astro::GalacticOrientation orientation = astro::orientation_km_2017):
        m_transform(
            orientation.gcs_to_reference_cs(),
            orientation.gcs_to_reference_cs()*(peculiar_velocity + la::Vector{0.0, circular_velocity, 0.0})) {};

    la::RigidTransform<double, 3> operator()([[maybe_unused]] double t) { return m_transform; }
    la::RigidTransform<double, 3> operator()() { return m_transform; }

private:
    la::RigidTransform<double, 3> m_transform;
};

class ECStoICRS
{
public:
    constexpr ECStoICRS() = default;

    [[nodiscard]] constexpr la::RigidTransform<double, 3>
    operator()([[maybe_unused]] double days_since_j2000) const noexcept { return transform(); }

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

// ICRS and BCRS share the same orientation and origin
using ECStoBCRS = ECStoICRS;

class ICRStoGCRS
{
public:
    constexpr ICRStoGCRS() = default;

    [[nodiscard]] la::RigidTransform<double, 3>
    operator()(double days_since_j2000) const noexcept
    {
        const la::Vector<double, 3> earth_velocity
            = astro::earth.orbit(days_since_j2000).reference_cs_velocity();
        return la::RigidTransform<double, 3>(
            la::RotationMatrix<double, 3>::identity(),
            ecs_to_icrs().rotation()*(-earth_velocity));
    }

private:
    static constexpr auto ecs_to_icrs = ECStoICRS{};
};

class GCRStoITRS
{
public:
    constexpr GCRStoITRS() = default;

    [[nodiscard]] la::RigidTransform<double, 3>
    operator()(double days_since_j2000) const noexcept
    {
        const double cip_x = astro::cip[0]((1.0/36525.0)*days_since_j2000);
        const double cip_y = astro::cip[1]((1.0/36525.0)*days_since_j2000);
        const double cip_r = std::hypot(cip_x, cip_y);
        const double cip_z = std::sqrt((1.0 + cip_r)*(1.0 - cip_r));
        const la::Vector<double, 3> cip = {cip_x, cip_y, cip_z};
        const auto align_cip = la::RotationMatrix<double, 3>::align_z(cip);

        // The polynomial deevelopment of the CIO locator is neglected here.
        const double cio_locator = -0.5*cip_x*cip_y;
        const double day_fraction = days_since_j2000 - std::floor(days_since_j2000);
        const auto to_tirs = la::RotationMatrix<double, 3>::coordinate_axis<Axis::z>(astro::earth.rotation_angle(day_fraction) - cio_locator);

        // Polar motion is neglected: ITRS = TIRS.
        return la::RigidTransform<double, 3>(to_tirs*align_cip, la::Vector<double, 3>{});
    }
};

class ITRStoHCS
{
public:
    constexpr ITRStoHCS() = default;
    constexpr ITRStoHCS(double longitude, double latitude):
        m_transform(transform(longitude, latitude)) {}

    [[nodiscard]] la::RigidTransform<double, 3>
    operator()([[maybe_unused]] double days_since_j2000) const noexcept { return m_transform; }

    [[nodiscard]] la::RigidTransform<double, 3>
    operator()() const noexcept { return m_transform; }

private:
    [[nodiscard]] static la::RigidTransform<double, 3>
    transform(double longitude, double latitude) noexcept
    {
        constexpr auto chaining = la::Chaining::intrinsic;
        const auto rotation = la::RotationMatrix<double, 3>::composite_axes<Axis::z, Axis::y, chaining>(std::numbers::pi + longitude, -latitude);

        const la::Vector<double, 3> translation = {0.0, -astro::earth.surface_speed(latitude), 0.0};

        return la::RigidTransform<double, 3>(rotation, translation);
    }

    la::RigidTransform<double, 3> m_transform;
};

} // namespace celestial

} // namespace zdm
