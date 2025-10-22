#pragma once

#include <numbers>

#include "matrix.hpp"
#include "polynomial.hpp"
#include "time.hpp"

namespace zdm
{

[[nodiscard]] constexpr double
mean_anomaly_from(double mean_longitude, double longitude_of_periapsis) noexcept
{
    return mean_longitude - longitude_of_periapsis;
}

[[nodiscard]] constexpr double mean_motion_from(double orbital_period_jd) noexcept
{
    return 2.0*std::numbers::pi/orbital_period_jd;
}

[[nodiscard]] inline double eccentric_anomaly_from(double eccentricity, double mean_anomaly) noexcept
{
    double ea = (eccentricity > 0.8) ? std::numbers::pi : mean_anomaly;
    for (std::size_t i = 0; i < 7; ++i)
        ea -= (ea - eccentricity*std::sin(ea) - mean_anomaly)/(1.0 - eccentricity*std::cos(ea));
    return ea;
}

[[nodiscard]] inline std::array<double, 2>
true_anomaly_cossin_from(double eccentricity, double eccentric_anomaly_cos, double eccentric_anomaly_sin)
{
    const double denom = 1.0/(1.0 - eccentricity*eccentric_anomaly_cos);
    return {
        (eccentric_anomaly_cos - eccentricity)*denom,
        std::sqrt((1.0 - eccentricity)*(1.0 + eccentricity))*eccentric_anomaly_sin*denom
    };
}

struct GalacticOrientation
{
    double ngp_dec;
    double ngp_ra;
    double ncp_lon;

    [[nodiscard]] inline la::RotationMatrix<double, 3>
    gcs_to_icrs() const noexcept
    {
        constexpr auto convention = la::EulerConvention::zyz;
        constexpr auto chaining = la::Chaining::intrinsic;

        return la::RotationMatrix<double, 3>::from_euler_angles<convention, chaining>(
                ncp_lon, 0.5*std::numbers::pi - ngp_dec, std::numbers::pi - ngp_ra);
    }
};

// Value of Karim and Mamajek (2017)
constexpr GalacticOrientation orientation_km_2017 = {
        27.084*std::numbers::pi/180.0,
        192.729*std::numbers::pi/180.0,
        122.928*std::numbers::pi/180.0
};

// Value of Schönrich, Binney, and Dehnen (2010)
constexpr la::Vector<double, 3> peculiar_velocity_sbd_2010 = {11.1, 12.24, 7.25};

struct OrbitOrientation
{
    double inclination;
    double longitude_of_the_ascending_node;
    double longitude_of_periapsis;

    [[nodiscard]] inline la::RotationMatrix<double, 3>
    orbital_plane_to_reference_cs() const noexcept
    {
        constexpr auto convention = la::EulerConvention::zxz;
        constexpr auto chaining = la::Chaining::intrinsic;

        const double argument_of_periapsis = longitude_of_periapsis - longitude_of_the_ascending_node;
        return la::RotationMatrix<double, 3>::from_euler_angles<convention, chaining>(
            -argument_of_periapsis, -inclination, -longitude_of_the_ascending_node);
    }
};

struct OrbitPosition
{
    double eccentricity;
    double semi_major_axis;
    double mean_longitude;
    double mean_motion;
};

class OrbitalState
{
public:
    constexpr OrbitalState() = default;
    OrbitalState(const OrbitOrientation& orientation, const OrbitPosition& position):
        m_orientation(orientation), m_position(position),
        m_orbital_plane_to_reference_cs(orientation.orbital_plane_to_reference_cs()) {}

    [[nodiscard]] constexpr double
    mean_anomaly() const noexcept
    {
        return m_position.mean_longitude - m_orientation.longitude_of_periapsis;
    }

    [[nodiscard]] inline double
    eccentric_anomaly() const noexcept
    {
        return eccentric_anomaly_from(m_position.eccentricity, mean_anomaly());
    }

    [[nodiscard]] inline la::Vector<double, 3>
    orbital_plane_velocity() const noexcept
    {
        const double ea = eccentric_anomaly_from(m_position.eccentricity, mean_anomaly());
        const double cos_ea = std::cos(ea);
        const double sin_ea = std::sin(ea);
        const auto& [cos_ta, sin_ta] = true_anomaly_cossin_from(m_position.eccentricity, cos_ea, sin_ea);
        const double speed = m_position.mean_motion*m_position.semi_major_axis/std::sqrt((1.0 - m_position.eccentricity)*(1.0 + m_position.eccentricity));
        return {-speed*sin_ta, speed*(m_position.eccentricity + cos_ta), 0.0};
    }

    [[nodiscard]] inline la::Vector<double, 3>
    reference_cs_velocity() const noexcept
    {
        return m_orbital_plane_to_reference_cs*orbital_plane_velocity();
    }

private:
    OrbitOrientation m_orientation;
    OrbitPosition m_position;
    la::RotationMatrix<double, 3> m_orbital_plane_to_reference_cs;
};

template<std::size_t N, std::size_t M, std::size_t P>
struct DynamicalOrbitOrientation
{
    Polynomial<double, N> inclination;
    Polynomial<double, M> longitude_of_the_ascending_node;
    Polynomial<double, P> longitude_of_periapsis;

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

template <std::size_t N, std::size_t M>
struct DynamicalKeplerOrbit
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

template <std::size_t N, std::size_t M, std::size_t P, std::size_t K, std::size_t L>
struct Orbit
{
    DynamicalOrbitOrientation<N, M, P> orientation;
    DynamicalKeplerOrbit<K, L> orbit;
    Time epoch;

    [[nodiscard]] constexpr OrbitalState
    operator()(double days_since_epoch) const noexcept
    {
        return OrbitalState(orientation(days_since_epoch), orbit(days_since_epoch));
    }
};

/**
    @brief Parameters of Earth's orbit.

    J. L. Simon, et al., “Numerical expressions for precession formulae and
    mean elements for the Moon and the planets.”, Astronomy and Astrophysics,
    vol. 282, EDP, p. 663, 1994.

    The semi-major axis is expressed in kilometers. The mean motion has units
    of radians per day. All the polynomials expect time in Julian millenia
    since the J2000 epoch. The polynomials corresponding to angle elements
    (inclination, longitude of the ascending node, longitude of periapsis, 
    mean longitude) give the angles in radians.
*/
constexpr Orbit earth_orbit = Orbit<4, 4, 4, 4, 4>{
    .orientation = {
        .inclination = Polynomial<double, 4>{
            2.2784928682258706e-03, -1.6243827829679335e-05,
           -5.9990844900493980e-07,  1.3089969389957470e-09},
        .longitude_of_the_ascending_node = Polynomial<double, 4>{
            3.0521126906052700e+00, -4.2078290028802140e-02,
            7.4379678623512010e-05,  2.5792087835027316e-08
        },
        .longitude_of_periapsis = Polynomial<double, 4>{
            1.7965956472674636e+00,  5.6298275557919950e-02,
            2.5828822167644984e-04, -6.8334488352389100e-07
        }
    },
    .orbit = {
        .eccentricity = Polynomial<double, 4>{
            1.6708634200000000e-02, -4.2036540000000000e-04,
           -1.2673400000000000e-05,  1.4440000000000000e-07
        },
        .mean_longitude = Polynomial<double, 4>{
            1.7534704594962450e+00,  6.2830758499914170e+03,
           -9.9101249369281360e-06, -2.5355755522028735e-08
        },
        .semi_major_axis = 1.495980229607128e+08,
        .mean_motion = 1.720212416151879e-02
    },
    .epoch = {
        .year = 2000, 
        .mon = 1,
        .mday = 1, 
        .hour = 12,
        .min = 0,
        .sec = 0
    }
};

} // namespace zdm 
