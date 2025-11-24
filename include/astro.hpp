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

#include <numbers>

#include "matrix.hpp"
#include "polynomial.hpp"
#include "time.hpp"

namespace zdm::astro
{

/**
    @brief Orientation of a galactic cooordinate system relative to a reference
    coordinate system.

    The orientation of a galactic coordinate system relative to a reference
    inertial coordinate system (typically ICRS) is completely specified by
    three angles:
        - declination of the north galactic pole (\f$\delta_\text{NGP}\f$),
        - right asecnsion of the north galactic pole (\f$\alpha_\text{NGP}\f$),
        - longitude of the z-axis (north celestial pole) of the reference
          coordinate system in the galactic coordinate system
          (\f$\ell_\text{NCP}\f$).
*/
struct GalacticOrientation
{

    double ngp_dec; /// Declination of the north galactic pole.
    double ngp_ra;  /// Right ascension of the north galactic pole.
    double ncp_lon; /// Galactic longitude of the north celestial pole.

    [[nodiscard]] constexpr bool operator==(const GalacticOrientation& other) const noexcept = default;

    /**
        @brief Rotation from the galactic coordinate system to the reference
        coordinate system.

        @return Rotation matrix from the galactic to the reference coordinate
        system.

        The rotation matrix from the galactic coordinate system to the
        reference coordinate system in terms of the orientation parameters
        \f$\delta_\text{NGP}\f$, \f$\alpha_\text{NGP}\f$, and
        \f$\ell_\text{NCP}\f$ is given by
        \f[
            R_Z(\pi - \alpha_\text{NGP})R_Y(\pi/2 - \delta_\text{NGP})R_Z(\ell_\text{NCP}).
        \f]

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
        - inclination of the orbit (\f$i\f$), defined as the angle of the
          orbital plane with the xy-plane of the reference coordinate system,
        - longitude of the ascending node (\f$\Omega\f$), defined as the angle
          from the reference x-axis of the point where the orbiting body
          crosses the reference plane in the positive reference z-direction,
        - longitude of the periapsis (\f$\varpi\f$), defined as the angle of
          the projection of the orbit's periapsis onto the reference xy-plane
          from the reference x-axis.
*/
struct OrbitOrientation
{
    double inclination;                     /// Inclination of the orbit.
    double longitude_of_the_ascending_node; /// Longitude of the ascending node of the orbit.
    double longitude_of_periapsis;          /// Longitude of periapsis of the orbit.

    [[nodiscard]] constexpr bool operator==(const OrbitOrientation& other) const noexcept = default;

    /**
        @brief Rotation from the orbital plane coordinate system to the
        reference coordinate system.

        @return Rotation matrix from the orbital coordinate system to the
        reference coordinate system.

        The orbital plane coordinate system is defined as a coordinate system
        whose z-axis is perpendicular to the orbital plane, and has a component
        parallel to the z-axis of the reference coordinate system, and whose
        x-axis is in the direction of the periapsis of the orbit.

        The rotation matrix from the orbital plane to the reference coordinate
        system in terms of the orientation parameters \f$i\f$, \f$\Omega\f$,
        and \f$\varpi\f$ is given by
        \f[
            R_Z(-\Omega)R_X(-i)R_Z(-\varpi).
        \f]
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
        - eccentricity of the orbit (\f$e\f$),
        - semi-major axis of the orbit (\f$a\f$),
        - mean longitude of the body along the orbit (\f$L\f$),
        - mean motion of the body (\f$n\f$)
*/
struct OrbitPosition
{
    double eccentricity;    /// Eccentricity of the orbit
    double semi_major_axis; /// Semi-major axis of the orbit in kilometers.
    double mean_longitude;  /// Mean longitude of the body.
    double mean_motion;     /// Mean motion of the body in radians per day.

    [[nodiscard]] constexpr bool operator==(const OrbitPosition& other) const noexcept = default;
};

/**
    @brief Instantaneous state of an orbiting body along Kepler orbit.

    Given any instance of time, the position of an orbiting body in space is
    completely determined by seven parameters. These parameters can be divided
    into two groups:
        - orbital orientation parameters, specifying the orientation of a Kepler
          orbit in space,
        - orbital position parameters, specifying the shape of the Kepler orbit
          and the position of the body along it.
*/
struct OrbitalState
{
    OrbitOrientation orientation;   /// Orientation of the orbit.
    OrbitPosition position;         /// Position of the body along the orbit.

    [[nodiscard]] constexpr bool operator==(const OrbitalState& other) const noexcept = default;

    /**
        @brief Mean anomaly of the body in radians.

        The mean anomaly \f$M\f$ of an elliptical orbit is an angle parameter
        which evolves at a constant rate. It can be computed as the difference
        of the mean longitude \f$L\f$ and the longitude of the periapsis
        \f$\varpi\f$ of the orbit: \f$M = L - \varpi\f$.
    */
    [[nodiscard]] constexpr double
    mean_anomaly() const noexcept
    {
        return position.mean_longitude - orientation.longitude_of_periapsis;
    }

    /**
        @brief Eccentric anomaly of the body in radians.

        The eccentric anomaly \f$E\f$ of the body is determined from the mean
        anomaly \f$M\f$ via Kepler's equation
        \f[
            M = E - e*sin(E),
        \f]
        where \f$e\f$ is the eccentricity of the orbit.
    */
    [[nodiscard]] double
    eccentric_anomaly() const noexcept
    {
        const double ma = mean_anomaly();

        // Mean anomaly to [-pi,pi]
        const double ma_mod = std::remainder(ma, 2.0*std::numbers::pi);

        // Use absolute value to solve Kepler's equation.
        const double ma_abs = std::fabs(ma_mod);

        // Initial guess:
        // Danby, J.M.A. The solution of Kepler's equation, III. Celestial
        // Mechanics 40, 303–312 (1987).
        double ea = ma_abs + 0.85*position.eccentricity;

        // 5 Newton iterations should be enough for any reasonable case.
        for (std::size_t i = 0; i < 5; ++i)
        {
            const double cos_ea = std::cos(ea);
            const double sin_ea = std::sin(ea);
            ea -= (ea - position.eccentricity*sin_ea - ma_abs)/(1.0 - position.eccentricity*cos_ea);
        }

        // restrict eccentric anomaly to [0,2pi]
        const double ea_mod = (ma_mod >= 0.0) ? ea : 2.0*std::numbers::pi - ea;

        assert(0 <= ea_mod && ea_mod <= 2.0*std::numbers::pi);
        return ea_mod;
    }

    /**
        @brief True anomaly of the body in radians.

        The true anomaly \f$\nu\f$ is the angle between the position of the
        body along the orbit and the periapsis as measured from the focus of
        the orbit. It is related to the eccentric anomaly \f$E\f$ by
        \f[
            \cos\nu = \frac{\cos E - e}{1 - e\cos E},
        \f]
        \f[
            \sin\nu = \frac{\sqrt{1 - e^2}\sin E}{1  e\cos E},
        \f]
        where \f$e\f$ is the eccentricity of the orbit.
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

        The velocity of a body at a point in its orbit in the orbital plane
        can be obtained from the true anomaly \f$\nu\f$
        \f[
            \vec{v} = \left(
                -na\frac{\sin\nu}{\sqrt{1 - e^2}},
                 na\frac{e + \cos\nu}{\sqrt{1 - e^2}}
            \right),
        \f]
        where \f$e\f$ is the eccentricity of the orbit, \f$n\f$ is the mean
        motion, and \f$a\f$ is the semi-major axis.

        @note The above equation defines a two-dimensional vector, but this
        function returns a three-dimensional vector whose z-component is set to
        zero.
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

        The velocity of the body in the reference coordinate system is obtained
        from the orbital plane velocity by rotating it with with the Euler
        angles given by the orientation parameters of the orbit.
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

    This structure determines the evolution of the orbital orientation parameter
    from a polynomial expansion for each of the parameters.

    The time evolution of classical orbital elements is typically given by
    short and long period variations, which lead to an evolution described
    on short time scales by a compbination of a polynomial expansion and
    trigonomertic terms describing shorter period variations. The trigonometric
    terms for Earth's orbit are on the order of arcseconds, and therefore
    the evolution here is described by only the mean, polynomial evolution.
*/
template<std::size_t N, std::size_t M, std::size_t P>
struct DynamicalOrbitOrientation
{
    Polynomial<double, N> inclination;
    Polynomial<double, M> longitude_of_the_ascending_node;
    Polynomial<double, P> longitude_of_periapsis;

    [[nodiscard]] constexpr bool operator==(const DynamicalOrbitOrientation& other) const noexcept = default;

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

    This structure determines the evolution of the shape (eccentricity and
    semi-major axis) and motion (mean longitude and mean motion) parameters of
    a Kepler orbit. The eccentricity and mean longitude are given by
    polynomials.

    The time evolution of classical orbital elements is typically given by
    short and long period variations, which lead to an evolution described
    on short time scales by a compbination of a polynomial expansion and
    trigonomertic terms describing shorter period variations. The trigonometric
    terms for Earth's orbit are on the order of arcseconds, and therefore
    the evolution here is described by only the mean, polynomial evolution.
*/
template <std::size_t N, std::size_t M>
struct KeplerOrbit
{
    Polynomial<double, N> eccentricity;
    Polynomial<double, M> mean_longitude;
    double semi_major_axis;
    double mean_motion;

    [[nodiscard]] constexpr bool operator==(const KeplerOrbit& other) const noexcept = default;

    /**
        @brief State of the body on the the orbit at a point in time.

        @param days_since_epoch Days since the epoch for which the evolution
        is defined.

        @return State of the body at the specified time.
    */
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

    [[nodiscard]] constexpr bool operator==(const Orbit& other) const noexcept = default;

    [[nodiscard]] constexpr OrbitalState
    operator()(double days_since_epoch) const noexcept
    {
        return {orientation(days_since_epoch), orbit(days_since_epoch)};
    }
};

template <std::size_t N, std::size_t M, std::size_t P, std::size_t K, std::size_t L>
Orbit(DynamicalOrbitOrientation<N, M, P>, KeplerOrbit<K, L>, time::Time) -> Orbit<N, M, P, K, L>;

/**
    @brief An oblate spheroid.

    An oblate spheroid is the surface of revolution formed by rotating an
    ellipse about its semi-minor axis. An oblate spheroid is fully specified
    by its flattening and equatorial radius, the latter corresponding to the
    semi-major axis of the ellipse that generates the spheriod.
*/
struct OblateSpheroid
{
    double flattening;
    double equatorial_radius;

    [[nodiscard]] constexpr bool operator==(const OblateSpheroid& other) const noexcept = default;
};

/**
    @brief A planet specified by its orbit, shape and rate of rotation.

    @tparam N Order of polynomial expansion for inclination.
    @tparam M Order of polynomial expansion for longitude of the ascending node.
    @tparam P Order of polynomial expansion for longitude of periapsis.
    @tparam K Order of polynomial expansion for eccentricity.
    @tparam L Order of polynomial expansion for mean longitude.

    This structure describes a planet as a rotating oblate spheroid on a
    perturbed elliptical orbit.
*/
template <std::size_t N, std::size_t M, std::size_t P, std::size_t K, std::size_t L>
struct Planet
{
    Orbit<N, M, P, K, L> orbit;
    OblateSpheroid spheroid;
    Polynomial<double, 1> rotation_angle;

    [[nodiscard]] constexpr bool operator==(const Planet& other) const noexcept = default;

    /**
        @brief Speed of a point on the planet's surface relative to its center.

        @param latitude Geodetic latitude of the point in radians \f$[-\pi/2,\pi/2]\f$.

        @return Speed of the point.

        The surface speed is given by \f$v = \omega\rho\f$, where \f$\omega\f$
        is the angular speed and \f$\rho\f$ is the the distance of the point
        from the axis of rotation. For a given latitude \f$\phi\f$, the
        distance \f$\rho = N\cos\phi\f$, where \f$N\f$ iis the prime vertical
        radius of curvature
        \f[
            N = \frac{a}{\sqrt{1 - e^2\sin^2\phi}},
        \f]
        where \f$a\f$ is the equatorial radius (semi-major axis) of the ellipse
        whose rotation about the axis of revolution generates the spheroid.
    */
    [[nodiscard]] double surface_speed(double latitude) const noexcept
    {
        assert(-0.5*std::numbers::pi <= latitude && latitude <= 0.5*std::numbers::pi);
        const double ecc_sq = spheroid.flattening*(2.0 - spheroid.flattening);
        const double sin_lat = std::sin(latitude);
        const double pvroc = spheroid.equatorial_radius/std::sqrt(1.0 - ecc_sq*sin_lat*sin_lat);
        return ((1.0/86400.0)*rotation_angle.derivative()(0.0))*pvroc;

    }
};

template <std::size_t N, std::size_t M, std::size_t P, std::size_t K, std::size_t L>
Planet(Orbit<N, M, P, K, L>, OblateSpheroid, Polynomial<double, 1>) -> Planet<N, M, P, K, L>;

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
    the purposes of this library.

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
    .spheroid = OblateSpheroid{
        .flattening = 3.352819697896193e-03,
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

} // namespace zdm::astro
