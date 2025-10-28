#pragma once

#include <numbers>

#include "matrix.hpp"
#include "polynomial.hpp"
#include "time.hpp"

namespace zdm
{

namespace astro
{

/**
    @brief Orientation of a galactic cooordinate system relative to a reference
    coordinate system.

    A galactic coordinate system at a given position in the galactic disk is
    determined by the north galactic pole (NGP), defined as the direction
    perpendicular to the plane of the galaxy, and the galactic center (GC). The
    orientation of the NGP is defined such that it has a component parallel to
    the z-axis of the reference coordinate.

    The z-axis of the galactic coordinate system is in the direction of the
    NGP, and the x-axis is in the direction of the projection of the GC onto
    the galactic plane. Namely, for positions close to the galactic plane, the
    z-axis is approximately the directon of the galactic center.

    The galactic coordinate system has no motion with respect to the galactic
    center. Namely, for a point revolving around the galactic center, whose
    reference coordinate system is attached to the point, its galactic
    coordinate system shares the origin of the reference coordinate system at
    every instance, but the coordinate systems have a relative velocity.

    The orientation of a galactic coordinate system relative to a reference
    inertial coordinate system (typically ICRS) is completely specified by
    three angles:
    - declination of the north galactic pole,
    - right asecnsion of the north galactic pole,
    - longitude of the z-axis (north celestial pole) of the reference
      coordinate system in the galactic coordinate system.
*/
struct GalacticOrientation
{

    double ngp_dec; /// Declination of the north galactic pole.
    double ngp_ra;  /// Right ascension of the north galactic pole.
    double ncp_lon; /// Galactic longitude of the north celestial pole.

    /**
        @brief Rotation from the galactic coordinate system to the reference
        coordinate system.

        @return Rotation matrix from the galactic to the reference coordinate
        system.

        @note This transformation rotates a vector from the galactic coordinate
        system to an intermediate coordinate system whose coordinate axes are
        aligned with those of the reference coordinate system, but which may
        still have some velocity relative to the reference coordinate system.
    */
    [[nodiscard]] la::RotationMatrix<double, 3>
    gcs_to_reference_cs() const noexcept
    {
        constexpr auto convention = la::EulerConvention::zyz;
        constexpr auto chaining = la::Chaining::intrinsic;

        return la::RotationMatrix<double, 3>::from_euler_angles<convention, chaining>(
                ncp_lon, 0.5*std::numbers::pi - ngp_dec, std::numbers::pi - ngp_ra);
    }
};

/**
    @brief Instantaneous orientation of a Kepler orbit in a reference
    coordinate system.

    The orientation of a Kepler orbit is completely determined by three angle
    parameters:
    - inclination of the orbit, defined as the angle of the orbital plane with
      the xy-plane of the reference coordinate system,
    - longitude of the ascending node, defined as the angle from the reference
      x-axis of the point where the orbiting body crosses the reference plane
      in the positive reference z-direction,
    - longitude of the periapsis, defined as the angle of the projection of
      the orbit's periapsis onto the reference xy-plane from the reference
      x-axis.
*/
struct OrbitOrientation
{
    double inclination;                     /// Inclination of the orbit.
    double longitude_of_the_ascending_node; /// Longitude of the ascending node of the orbit.
    double longitude_of_periapsis;          /// Longitude of periapsis of the orbit.

    /**
        @brief Rotation from the orbital plane coordinate system to the
        reference coordinate system.

        @return Rotation matrix from the orbital coordinate system to the
        reference coordinate system.

        The orbital plane coordinate system is defined as a coordinate system
        whose z-axis is perpendicular to the orbital plane, and has a component
        parallel to the z-axis of the reference coordinate system, and whose
        x-axis is in the direction of the periapsis of the orbit.

        The orbital plane coordinate system shares the origin and is stationary
        relative to the reference coordinate system.
    */
    [[nodiscard]] la::RotationMatrix<double, 3>
    orbital_plane_to_reference_cs() const noexcept
    {
        constexpr auto convention = la::EulerConvention::zxz;
        constexpr auto chaining = la::Chaining::intrinsic;

        const double argument_of_periapsis = longitude_of_periapsis - longitude_of_the_ascending_node;
        return la::RotationMatrix<double, 3>::from_euler_angles<convention, chaining>(
            -argument_of_periapsis, -inclination, -longitude_of_the_ascending_node);
    }
};

/**
    @brief Instantaneous position of a body along a Kepler orbit.

    The position and velocity of a body along a Kepler orbit are completely
    determined by four parameters:
    - eccentricity of the orbit,
    - semi-major axis of the orbit,
    - mean longitude of the body along the orbit,
    - mean motion of the body
*/
struct OrbitPosition
{
    double eccentricity;    /// Eccentricity of the orbit
    double semi_major_axis; /// Semi-major axis of the orbit in kilometers.
    double mean_longitude;  /// Mean longitude of the body.
    double mean_motion;     /// Mean motion of the body in radians per day.
};

/**
    @brief Instantaneous state of an orbiting body along Kepler orbit.

    Given any instance of time, the position of an orbiting body in space is
    completely determined by seven parameters. These parameters can be divided
    into two groups:
    - orbital orientation parameters, specifying the orientation of a Kepler
      orbit in space,
    - orbital position parameters, specifying the shape of the Kepler orbit and
      the position of the body along it.
*/
struct OrbitalState
{
    OrbitOrientation orientation;   /// Orientation of the orbit.
    OrbitPosition position;         /// Position of the body along the orbit.

    /**
        @brief Mean anomaly of the body in radians.

        @return Mean anomaly of the body.

        The mean anomaly of an elliptical orbit is an angle parameter which
        evolves at a constant rate. It can be computed as the difference of the
        mean longitude and the longitude of the periapsis of the orbit.
    */
    [[nodiscard]] constexpr double
    mean_anomaly() const noexcept
    {
        return position.mean_longitude - orientation.longitude_of_periapsis;
    }

    /**
        @brief Eccentric anomaly of the body in radians.

        @return Eccentric anomaly of the body.

        The eccentric anomaly `E` of the body is determined from the mean
        anomaly `M` via Kepler's equation `M = E - e*sin(E)`, where `e` is the
        eccentricity of the orbit.
    */
    [[nodiscard]] double
    eccentric_anomaly() const noexcept
    {
        const double ma = mean_anomaly();
        double ea = (position.eccentricity > 0.8) ? std::numbers::pi : ma;
        for (std::size_t i = 0; i < 7; ++i)
            ea -= (ea - position.eccentricity*std::sin(ea) - ma)/(1.0 - position.eccentricity*std::cos(ea));
        return ea;
    }

    /**
        @brief True anomaly of the body in radians.

        @brief True anomaly of the body.

        The true anomaly is the angle between the position of the body along
        the orbit and the periapsis as measured from the focus of the orbit.
    */
    [[nodiscard]] double
    true_anomaly() const noexcept
    {
        const double ea = eccentric_anomaly();
        const double cos_ea = std::cos(ea);
        const double sin_ea = std::sin(ea);
        const auto& [cos_ta, sin_ta] = true_anomaly_cossin(cos_ea, sin_ea);
        return std::atan2(sin_ta, cos_ta);
    }

    /**
        @brief Velocity of the body in the orbital plane in km/s.

        @brief Velocity of the body.
    */
    [[nodiscard]] la::Vector<double, 3>
    orbital_plane_velocity() const noexcept
    {
        const double ea = eccentric_anomaly();
        const double cos_ea = std::cos(ea);
        const double sin_ea = std::sin(ea);
        const auto& [cos_ta, sin_ta] = true_anomaly_cossin(cos_ea, sin_ea);
        const double speed = 86400.0*position.mean_motion*position.semi_major_axis/std::sqrt((1.0 - position.eccentricity)*(1.0 + position.eccentricity));
        return {-speed*sin_ta, speed*(position.eccentricity + cos_ta), 0.0};
    }

    /**
        @brief Velocity of the body in the reference coordinate system in km/s.

        @brief Velocity of the body.
    */
    [[nodiscard]] la::Vector<double, 3>
    reference_cs_velocity() const noexcept
    {
        return orientation.orbital_plane_to_reference_cs()*orbital_plane_velocity();
    }

private:
    [[nodiscard]] std::array<double, 2>
    true_anomaly_cossin(double ea_cos, double ea_sin) const noexcept
    {
        const double denom = 1.0/(1.0 - position.eccentricity*ea_cos);
        return {
            (ea_cos - position.eccentricity)*denom,
            std::sqrt((1.0 - position.eccentricity)*(1.0 + position.eccentricity))*ea_sin*denom
        };
    }
};

/**
    @brief Time evolution of the mean orientation parameters of an orbit.

    @tparam N Order of polynomial expansion for inclination.
    @tparam M Order of polynomial expansion for longitude of the ascending node.
    @tparam P Order of polynomial expansion for longitude of periapsis.

    The time evolution of classical orbital elements is typically given by
    short and long period variations, which lead to an evolution described
    on short time scales by a compbination of a polynomial expansion and
    trigonomertic terms describing shorter period variations. The trigonometric
    terms for Earth's orbit are on the order of arcseconds, and therefore
    the evolution here is described by only the mean, polynomial evolution.

    References:
    -   J. L. Simon, et al., “Numerical expressions for precession formulae and
        mean elements for the Moon and the planets.”, Astronomy and
        Astrophysics, vol. 282, EDP, p. 663, 1994.
*/
template<std::size_t N, std::size_t M, std::size_t P>
struct DynamicalOrbitOrientation
{
    Polynomial<double, N> inclination;
    Polynomial<double, M> longitude_of_the_ascending_node;
    Polynomial<double, P> longitude_of_periapsis;

    /**
        @brief Orientation of the orbit at a point in time.

        @param days_since_epoch Days since the epoch for which the evolution
        is defined.

        @return Orientation of the orbit at the specified time.
    */
    [[nodiscard]] constexpr OrbitOrientation
    operator()(double days_since_epoch) const noexcept
    {
        const double millenia_since_epoch = (1.0/365250.0)*days_since_epoch;
        return {
            inclination(millenia_since_epoch),
            longitude_of_the_ascending_node(millenia_since_epoch),
            longitude_of_periapsis(millenia_since_epoch)
        };
    }
};

template <std::size_t N, std::size_t M, std::size_t P>
DynamicalOrbitOrientation(Polynomial<double, N>, Polynomial<double, M>, Polynomial<double, P>) -> DynamicalOrbitOrientation<N, M, P>;

/**
    @brief Time evolution of the shape and motion parameters of an orbit.

    @tparam N Order of polynomial expansion for eccentricity.
    @tparam M Order of polynomial expansion for mean longitude.

    The time evolution of classical orbital elements is typically given by
    short and long period variations, which lead to an evolution described
    on short time scales by a compbination of a polynomial expansion and
    trigonomertic terms describing shorter period variations. The trigonometric
    terms for Earth's orbit are on the order of arcseconds, and therefore
    the evolution here is described by only the mean, polynomial evolution.

    References:
    -   J. L. Simon, et al., “Numerical expressions for precession formulae and
        mean elements for the Moon and the planets.”, Astronomy and
        Astrophysics, vol. 282, EDP, p. 663, 1994.
*/
template <std::size_t N, std::size_t M>
struct KeplerOrbit
{
    Polynomial<double, N> eccentricity;
    Polynomial<double, M> mean_longitude;
    double semi_major_axis;
    double mean_motion;

    [[nodiscard]] constexpr OrbitPosition
    operator()(double days_since_epoch) const noexcept
    {
        const double millenia_since_epoch = (1.0/365250.0)*days_since_epoch;
        return {
            eccentricity(millenia_since_epoch),
            semi_major_axis,
            mean_longitude(millenia_since_epoch),
            mean_motion
        };
    }
};

template <std::size_t N, std::size_t M>
KeplerOrbit(Polynomial<double, N>, Polynomial<double, M>, double, double) -> KeplerOrbit<N, N>;

/**
    @brief Orbit of a celestial body.

    @tparam N Order of polynomial expansion for inclination.
    @tparam M Order of polynomial expansion for longitude of the ascending node.
    @tparam P Order of polynomial expansion for longitude of periapsis.
    @tparam K Order of polynomial expansion for eccentricity.
    @tparam L Order of polynomial expansion for mean longitude.

    The state of an body on an elliptical orbit is specified by the orientation
    and shape of the orbit, in conjunction with the position and motion of the
    body along the orbit. For a perturbed orbit, both the orientation and shape
    of the orbit vary over time.

    This structure describes the evolution of a perturbed orbit over time using
    a parametrization of the orbit in terms of the orientaton parameters
    (inclination, longitude of the ascending node, and longitude of
    perihelion), shape parameters (eccentricity and semi-major axis), the
    position (mean longitude) of the body along the orbit, and the speed (mean
    motion) of the body.

    The evolution of the orbital elements due to perturbations is descbribed
    here purely by polyunomial expressions, leaving out short period
    variations, which for relevant orbits (namely, the Earth's) are on the
    order of arc seconds, which is less than the accuracy required for the
    purposes of this library.
*/
template <std::size_t N, std::size_t M, std::size_t P, std::size_t K, std::size_t L>
struct Orbit
{
    DynamicalOrbitOrientation<N, M, P> orientation;
    KeplerOrbit<K, L> orbit;
    time::Time epoch;

    [[nodiscard]] constexpr OrbitalState
    operator()(double days_since_epoch) const noexcept
    {
        return {orientation(days_since_epoch), orbit(days_since_epoch)};
    }
};

template <std::size_t N, std::size_t M, std::size_t P, std::size_t K, std::size_t L>
Orbit(DynamicalOrbitOrientation<N, M, P>, KeplerOrbit<K, L>, time::Time) -> Orbit<N, M, P, K, L>;

struct Ellipsoid
{
    double inverse_flattening;
    double equatorial_radius;
};

template <std::size_t N, std::size_t M, std::size_t P, std::size_t K, std::size_t L>
struct Planet
{
    Orbit<N, M, P, K, L> orbit;
    Ellipsoid ellipsoid;
    Polynomial<double, 1> rotation_angle;

    [[nodiscard]] double surface_speed(double latitude) const noexcept
    {
        const double ecc_sq = 2.0*(ellipsoid.inverse_flattening - 1.0)/(ellipsoid.inverse_flattening*ellipsoid.inverse_flattening);
        const double sin_lat = std::sin(latitude);
        const double pvroc = ellipsoid.equatorial_radius/std::sqrt(1.0 - ecc_sq*sin_lat*sin_lat);
        return ((1.0/86400.0)*rotation_angle.derivative()(0.0))*pvroc;

    }
};

template <std::size_t N, std::size_t M, std::size_t P, std::size_t K, std::size_t L>
Planet(Orbit<N, M, P, K, L>, Ellipsoid, Polynomial<double, 1>) -> Planet<N, M, P, K, L>;

/**
    @brief Orientation parameters of the galactic coordinate system according
    to Karim and Mamajek (2017).

    References:
    -   Karim, T. and Mamajek, E. E., “Revised geometric estimates of the North
        Galactic Pole and the Sun's height above the Galactic mid-plane”,
        Monthly Notices of the Royal Astronomical Society, vol. 465, no. 1,
        OUP, pp. 472–481, 2017. doi:10.1093/mnras/stw2772.
*/
constexpr GalacticOrientation orientation_km_2017 = {
    .ngp_dec = 27.084*std::numbers::pi/180.0,
    .ngp_ra = 192.729*std::numbers::pi/180.0,
    .ncp_lon = 122.928*std::numbers::pi/180.0
};

/**
    @brief Peculiar velocity of the solar system accordinh to Schönrich, Binney
    and Dehnen (2010).

    References:
    -   Schönrich, R., Binney, J., and Dehnen, W., “Local kinematics and the
        local standard of rest”, Monthly Notices of the Royal Astronomical
        Society, vol. 403, no. 4, OUP, pp. 1829–1833, 2010.
        doi:10.1111/j.1365-2966.2010.16253.x.
*/
constexpr la::Vector peculiar_velocity_sbd_2010 = {11.1, 12.24, 7.25};

/**
    @brief Constant parameters defining Earth.

    The unit conventions of the parameters are defined as follows.

    For the orbital parameters, the semi-major axis is expressed in kilometers.
    The mean motion has units of radians per day. All the polynomials expect
    time in Julian millenia since the J2000 epoch. The polynomials for the
    angle elements (inclination, longitude of the ascending node, longitude of
    periapsis, mean longitude) give the angles in radians.

    The rotation angle is defined in radians, and expects time in UT1 days
    since the J2000 epoch.

    The polynomials for the orbital parameters are truncated after the fourth
    term, because the added precision from further terms is unnecessary for
    the purposes of this library, and truncating to four terms leads to good
    memory alignment: one polynomials fits in an AVX2 register, two fit on a
    cache line.

    Refenrences:
    -   J. L. Simon, et al., “Numerical expressions for precession formulae and
        mean elements for the Moon and the planets.”, Astronomy and
        Astrophysics, vol. 282, EDP, p. 663, 1994.
    -   Capitaine, N., Guinot, B., and McCarthy, D. D., 2000, “Definition of
        the Celestial Ephemeris origin and of UT1 in the International
        Celestial Reference Frame,” Astron. Astrophys., 355(1), pp. 398–405.
*/
constexpr Planet earth = {
    .orbit = Orbit{
        .orientation = DynamicalOrbitOrientation{
            .inclination = Polynomial{
                2.2784928682258706e-03, -1.6243827829679335e-05,
               -5.9990844900493980e-07,  1.3089969389957470e-09},
            .longitude_of_the_ascending_node = Polynomial{
                3.0521126906052700e+00, -4.2078290028802140e-02,
                7.4379678623512010e-05,  2.5792087835027316e-08
            },
            .longitude_of_periapsis = Polynomial{
                1.7965956472674636e+00,  5.6298275557919950e-02,
                2.5828822167644984e-04, -6.8334488352389100e-07
            }
        },
        .orbit = KeplerOrbit{
            .eccentricity = Polynomial{
                1.6708634200000000e-02, -4.2036540000000000e-04,
               -1.2673400000000000e-05,  1.4440000000000000e-07
            },
            .mean_longitude = Polynomial{
                1.7534704594962450e+00,  6.2830758499914170e+03,
               -9.9101249369281360e-06, -2.5355755522028735e-08
            },
            .semi_major_axis = 1.495980229607128e+08,
            .mean_motion = 1.720212416151879e-02
        },
        .epoch = time::j2000_utc
    },
    .ellipsoid = Ellipsoid{
        .inverse_flattening = 298.25642,
        .equatorial_radius = 6.3781366e+03,
    },
    .rotation_angle = Polynomial{4.894961212823756, 0.01720217957524373}
};
/**
    @brief X and Y coordinates of the celestial intermediate pole.

    The units of the polynomials have been translated to radians, and they
    expect time in Julian centuries since the J200 epoch.

    This pair of polynomials describes the polynomial part of the development
    of the celestial intermediate pole based on the P03 precession model. The
    periodic variations are neglected on the basis that their contributions
    always remain on the order of arcseconds, far beyond the angular resolution
    needs of this library. They give little benefit for much more computation.
    For the same reason, the polynomials have been truncated to four terms for
    optimal memory alignment.

    References:
    -   Capitaine, N. and Wallace, P. T., 2006, “High precision methods for
        locating the celestial intermediate pole and origin,” Astron.
        Astrophys., 450, pp. 855–872, doi:10.1051/0004-6361:20054550.8-989-6
*/
constexpr std::array<Polynomial<double, 3>, 2> cip = {
    Polynomial{
        -8.0561489389971590e-08,  9.7165965171928760e-03,
        -2.0836462982693160e-06, -9.6292888551265400e-07
    },
    Polynomial{
        -3.3699398973923850e-08, -1.2554735086012543e-07,
        -1.0863353330939572e-04,  9.2143203417997300e-09
    }
};

} // namespace astro

} // namespace zdm
