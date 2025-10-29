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

template <typename T, typename TransformType>
concept parametric_transform_on = requires (T x, typename TransformType::value_type t)
    {
        { x(t) } -> std::same_as<TransformType>;
    };

template <typename T>
concept noop_transform = parametric_transform_on<T, void>;

template <typename T, typename... TransformTypes>
concept parametric_transform_on_one_of = (parametric_transform_on<T, TransformTypes> || ...);

template <typename T, typename RigidTransformType>
concept parametric_rigid_transform
    = parametric_transform_on_one_of<T,
        RigidTransformType,
        typename RigidTransformType::rotation_matrix_type,
        typename RigidTransformType::vector_type>;

/**
    @brief A helper for writing efficient compositions of parametric rigid
    transforms.

    @tparam T value type of the rigid transform.
    @tparam N dimension of the rigid transform.
    @tparam Types types of the composed rigid transforms.

    This helper class may be used to compose together any combination of
    objects which qualify as a parametric rigid transform. Specifically,
    given the value type `T` and a dimension `N`, if a type `U` defines a
    call operator `operator()`, which takes an argument of type `T`, and
    returns either a `RigidTransform<T, N>`, `RotationMatrix<T, N>`,
    `Vector<T, N>`, or `void`, then it qualifies as a parametric rigid
    transform.

    The special types with return type `void` represent identity transforms.
    This exists for explictly describing transforms in composition chains,
    which are in principle present, but in practice reduce to an identity
    operation.

    The compositon here is to be understood in a left to right order. That is,
    if we have a chain of transformations `T1 -> T2 -> ... -> TN`, then the
    transformations are given in the order `T1, T2, ..., TN`.
*/
template <std::floating_point T, std::size_t N, typename... Types>
    requires (parametric_rigid_transform<Types, la::RigidTransform<T, N>> && ...)
class CompositeRigidTransform
{
public:
    using rigid_transform_type = la::RigidTransform<T, N>;

    CompositeRigidTransform(const Types&... transforms): m_transforms(transforms...) {}

    /**
        @brief Call operator of the parametric rigid transform.

        @param parameter Parameter of the transform.

        @return Rigid transform at the given parameter value.
    */
    [[nodiscard]] rigid_transform_type
    operator()(const rigid_transform_type::value_type& parameter) const noexcept
    {
        return std::apply([&](Types... transforms)
        {
            auto res = rigid_transform_type::identity();
            ([&]{
                if constexpr (!noop_transform<Types>)
                    res = la::compose(res, transforms(parameter));
            }(), ...);
            return res;
        }, m_transforms);
    }

private:
    std::tuple<Types...> m_transforms;
};

/**
    @brief Transformation from the Galactic Coordinate System (GCS) to the
    International Celestial Reference System (ICRS).

    This class represents the transformation from the Galactic Coordinate
    System (GCS) to the International Celestial Reference System (ICRS).

    The galactic coordinate system is determined by the north galactic pole
    (NGP), defined as the direction perpendicular to the plane of the galaxy,
    and the galactic center (GC). The orientation of the NGP is defined such
    that it has a component parallel to the z-axis of the reference coordinate.

    The z-axis of the galactic coordinate system is in the direction of the
    NGP, and the x-axis is in the direction of the projection of the GC onto
    the galactic plane. Namely, for positions close to the galactic plane, the
    z-axis is approximately the directon of the galactic center.

    The galactic coordinate system has no motion with respect to the galactic
    center. This means that it has a relative velocity to the ICRS, given by
    the inverse of the sum of the local circular velocity and peculiar velocity
    of the solar system.

    The ICRS is the standard reference celestial reference system defined to be
    nonrotating and centered at the barycenter of the solar system. By
    convention, its orientation approximately matches that of the traditonally
    used equatorial coordinate system at the J2000 epoch.

    References:
    -   IERS Conventions (2010). Gérard Petit and Brian Luzum (eds.). (IERS
        Technical Note ; 36) Frankfurt am Main: Verlag des Bundesamts für
        Kartographie und Geodäsie, 2010. 179 pp., ISBN 3-89888-989-6
*/
class GCStoICRS
{
public:
    GCStoICRS(
        double circular_velocity,
        const la::Vector<double, 3>& peculiar_velocity = astro::peculiar_velocity_sbd_2010,
        const astro::GalacticOrientation& orientation = astro::orientation_km_2017):
        m_transform(
            orientation.gcs_to_reference_cs(),
            orientation.gcs_to_reference_cs()*(peculiar_velocity + la::Vector{0.0, circular_velocity, 0.0})) {};

    /**
        @brief Call operator of the parametric transform.

        @param days_since_j2000 Number of 24-hour days since the J2000 epoch.

        @return GCS to ICRS transform at the given time.

        @note Since the GCS and ICRS are both defined to be inertial coordinate
        systems, the transformation between them has no actual time dependence.
        This call operator just returns the same `RigidTransform` as the one
        that takes no parameters.
    */
    [[nodiscard]] constexpr la::RigidTransform<double, 3>
    operator()([[maybe_unused]] double days_since_j2000) const noexcept { return m_transform; }

    /**
        @brief Call operator of the constant transform.

        @return GCS to ICRS transform.
    */
    [[nodiscard]] constexpr la::RigidTransform<double, 3>
    operator()() const noexcept { return m_transform; }

private:
    la::RigidTransform<double, 3> m_transform;
};

/**
    @brief Transformation from the Ecliptic Coordinate System (ECS) to the
    International Celestial Reference System (ICRS).

    This class represents a transformation from the Ecliptic Coordinate System
    (ECS) to the International Celestial Reference System (ICRS).

    For simplicity, the ecliptic coordinate system here is defined to match the
    ecliptic of the J2000 epoch. This allows the transformation to between the
    coordinate systems to be time-independent. This means that the tilt of its
    equatorial plane relative to the equatorial plane of the ICRS is given by
    the J2000 value of the obliquity, and its x-axis is given by the J2000 
    direction of the vernal equinox.

    The ICRS is the standard reference celestial reference system defined to be
    nonrotating and centered at the barycenter of the solar system. By
    convention, its orientation approximately matches that of the traditonally
    used equatorial coordinate system at the J2000 epoch.

    References:
    -   IERS Conventions (2010). Gérard Petit and Brian Luzum (eds.). (IERS
        Technical Note ; 36) Frankfurt am Main: Verlag des Bundesamts für
        Kartographie und Geodäsie, 2010. 179 pp., ISBN 3-89888-989-6
*/
class ECStoICRS
{
public:
    constexpr ECStoICRS() = default;

    /**
        @brief Call operator of the parametric transform.

        @param days_since_j2000 Number of 24-hour days since the J2000 epoch.

        @return ECS to ICRS transform at the given time.

        @note Since the ECS and ICRS are both defined to be inertial coordinate
        systems, the transformation between them has no actual time dependence.
        This call operator just returns the same `RigidTransform` as the one
        that takes no parameters.
    */
    [[nodiscard]] constexpr la::RotationMatrix<double, 3>
    operator()([[maybe_unused]] double days_since_j2000) const noexcept { return transform(); }

    /**
        @brief Call operator of the constant transform.

        @return ECS to ICRS transform.
    */
    [[nodiscard]] constexpr la::RotationMatrix<double, 3>
    operator()() const noexcept { return transform(); }

private:
    // This function can be consteval because the value of the J2000 obliquity
    // for the purposes of this program is fixed from the IAU 2009 System of
    // Astronomical Constants at 84381.406 arc seconds. This code can be
    // updated if a better estimate is ever adopted by the IAU. Not that it
    // matters, because this code really doesn't require mas level accuracy.
    [[nodiscard]] static constexpr la::RotationMatrix<double, 3>
    transform() noexcept
    {
        // Cosine and sine of the J2000 obliquity of the ecliptic.
        constexpr double cos_obliquity_j2000 = 0.9174821430652418;
        constexpr double sin_obliquity_j2000 = 0.4090926006005829;
        constexpr auto res = la::RotationMatrix<double, 3>({
                1.0,  0.0,                 0.0,
                0.0,  cos_obliquity_j2000, sin_obliquity_j2000,
                0.0, -sin_obliquity_j2000, cos_obliquity_j2000,
            });
        return res;
    }
};

// ICRS and BCRS share the same orientation and origin
/**
    @brief Transformation from the Ecliptic Coordinate System (ECS) to the
    Barycentric Celestial Reference System (BCRS).

    The Barycentric Celestial Reference System is definitionally equivalent
    to the International Celestial Reference System. Therefore `ECStoBCRS` is
    just an alias to `ECStoICRS`. For further documentation, see the
    documentation of `ECStoICRS`.
*/
using ECStoBCRS = ECStoICRS;

/**
    @brief Transformation from the International Celestial Reference System
    (ICRS) to the Geocentric Celestial Reference System (GCRS).

    The Geocentric Celestial Reference System (GCRS) is essentially a
    translated variation on the International Celestial Reference System
    (ICRS). Specifically, they share the same orientation, but the ICRS has
    its origin at the solar system barycenter, whereas the GCRS has its origin
    at Earth's geocenter. This also means that the GCRS is boosted by Earth's
    orbital velocity with respect to the ICRS.

    The ICRS is the standard reference celestial reference system defined to be
    nonrotating and centered at the barycenter of the solar system. By
    convention, its orientation approximately matches that of the traditonally
    used equatorial coordinate system at the J2000 epoch.

    References:
    -   IERS Conventions (2010). Gérard Petit and Brian Luzum (eds.). (IERS
        Technical Note ; 36) Frankfurt am Main: Verlag des Bundesamts für
        Kartographie und Geodäsie, 2010. 179 pp., ISBN 3-89888-989-6
*/
class ICRStoGCRS
{
public:
    constexpr ICRStoGCRS() = default;

    /**
        @brief Call operator of the parametric transform.

        @param days_since_j2000 Number of 24-hour days since the J2000 epoch.

        @return ICRS to GCRS transform at the given time.
    */
    [[nodiscard]] la::Vector<double, 3>
    operator()(double days_since_j2000) const noexcept
    {
        const la::Vector<double, 3> earth_velocity
            = astro::earth.orbit(days_since_j2000).reference_cs_velocity();
        return ecs_to_icrs()*(-earth_velocity);
    }

private:
    static constexpr auto ecs_to_icrs = ECStoICRS{};
};

/**
    @brief Transformation from the Geocentric Celestial Reference System (GCRS)
    to the Celestial Intermediate Reference System (CIRS).

    The Celestial Intermediate Reference System is an intermediate coordinate
    system in the transformation from the GCRS to the International Terrestrial
    Reference System (ITRS), which defines the standard coordinates for
    Earth-based observations. The GCRS to ITRS transformation is broken into
    three parts to separate the celestial precession and nutation of the ITRS
    pole in the GCRS due to gravitational effects and the polar motion due to
    terrestrial effects. This leads to the definiton of the Celestial
    Intermediate Pole (CIP), which is the pole of the intermediate coordinate
    systems. Thus, the CIRS is the intermediate coordinate system which has CIP
    as its pole, but does not rotate along with the Earth.

    The GCRS is a geocentric variaton of the International Celestial Reference
    System (ICRS), which has the geocenter of Earth as its origin, in contrast
    to the ICRS, which is placed at the solar system barycenter. The ICRS is
    the standard reference celestial reference system defined to be nonrotating
    and centered at the barycenter of the solar system. By convention, its
    orientation approximately matches that of the traditonally used equatorial
    coordinate system at the J2000 epoch.

    References:
    -   IERS Conventions (2010). Gérard Petit and Brian Luzum (eds.). (IERS
        Technical Note ; 36) Frankfurt am Main: Verlag des Bundesamts für
        Kartographie und Geodäsie, 2010. 179 pp., ISBN 3-89888-989-6
*/
class GCRStoCIRS
{
public:
    constexpr GCRStoCIRS() = default;

    [[nodiscard]] la::RotationMatrix<double, 3>
    operator()(double days_since_j2000) const noexcept
    {
        const double cip_x = astro::cip[0]((1.0/36525.0)*days_since_j2000);
        const double cip_y = astro::cip[1]((1.0/36525.0)*days_since_j2000);
        const double cip_r = std::hypot(cip_x, cip_y);
        const double cip_z = std::sqrt((1.0 + cip_r)*(1.0 - cip_r));
        const la::Vector<double, 3> cip = {cip_x, cip_y, cip_z};
        const double cio_locator = -0.5*cip_x*cip_y;
        return la::RotationMatrix<double, 3>::coordinate_axis<Axis::z>(-cio_locator)*la::RotationMatrix<double, 3>::align_z(cip);
    }
};

class CIRStoTIRS
{
public:
    constexpr CIRStoTIRS() = default;

    [[nodiscard]] la::RotationMatrix<double, 3>
    operator()(double days_since_j2000) const noexcept
    {
        const double day_fraction = days_since_j2000 - std::floor(days_since_j2000);
        return la::RotationMatrix<double, 3>::coordinate_axis<Axis::z>(astro::earth.rotation_angle(day_fraction));
    }
};

class TIRStoITRS
{
public:
    void operator()([[maybe_unused]] double days_since_j2000) {}
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
