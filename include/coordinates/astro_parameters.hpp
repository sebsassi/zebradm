#pragma once

#include <array>
#include <cmath>
#include <numbers>

#include "numerical/polynomial.hpp"
#include "time.hpp"
#include "numerical/linalg.hpp"

namespace coordinates
{

class GalacticParameters
{
public:
    constexpr GalacticParameters(
        Epoch p_epoch, double p_ngp_ra, double p_ngp_dec, double p_ncp_lon)
    : m_epoch(p_epoch), m_ngp_ra(p_ngp_ra), m_ngp_dec(p_ngp_dec),
    m_ncp_lon(p_ncp_lon) {}

    [[nodiscard]] constexpr
    Matrix<double, 3, 3> rotation_to_equ() const noexcept
    {
        const double cal = std::cos(m_ngp_ra - std::numbers::pi);
        const double cdel = std::cos(m_ngp_dec - 0.5*std::numbers::pi);
        const double cel = std::cos(m_ncp_lon);

        const double sal = std::sin(m_ngp_ra - std::numbers::pi);
        const double sdel = std::sin(m_ngp_dec - 0.5*std::numbers::pi);
        const double sel = std::sin(m_ncp_lon);
        return {
            Vector<double, 3>{cal*cdel*cel + sal*sel, cal*cdel*sel - sal*cel, cal*sdel},
            Vector<double, 3>{sal*cdel*cel - cal*sel, sal*cdel*sel + cal*cel, sal*sdel},
            Vector<double, 3>{-sdel*cel, -sdel*sel, cdel}
        };
    }

private:
    Epoch m_epoch;
    double m_ngp_ra;
    double m_ngp_dec;
    double m_ncp_lon;
};

class Orbit
{
public:
    constexpr Orbit(
        Epoch p_epoch, double p_epoch_semimajor_axis,
        double p_epoch_eccentricity, double p_epoch_mean_anomaly,
        double p_epoch_mean_speed, Polynomial<double, 3> p_obliquity, 
        Polynomial<double, 3> p_ecliptic_longitude)
    : m_epoch(p_epoch), m_epoch_semimajor_axis(p_epoch_semimajor_axis), 
    m_epoch_eccentricity(p_epoch_eccentricity),
    m_epoch_mean_anomaly(p_epoch_mean_anomaly),
    m_epoch_mean_speed(p_epoch_mean_speed),
    m_axis_ratio(std::sqrt(1.0 - p_epoch_eccentricity*p_epoch_eccentricity)),
    m_obliquity(p_obliquity), 
    m_ecliptic_longitude(p_ecliptic_longitude)
    {
        m_speed_param = m_epoch_mean_speed*m_epoch_semimajor_axis*(1.0/86400.0)/m_axis_ratio;
    }

    [[nodiscard]] constexpr Vector<double, 3> orbital_velocity(double time) const noexcept
    {
        const double ea = eccentric_anomaly(time);
        const double cos_ea = std::cos(ea);
        const double sin_ea = std::sin(ea);

        const double ta = ta_from_ea(
                cos_ea, sin_ea, m_epoch_eccentricity, m_axis_ratio);
        const double cos_ta = std::cos(ta);
        const double sin_ta = std::sin(ta);
        
        const Vector<double, 2> velocity = {
            -m_speed_param*sin_ta, m_speed_param*(m_epoch_eccentricity + cos_ta)
        };

        const double nu = ecliptic_longitude(time);
        const double cos_nu = std::cos(nu);
        const double sin_nu = std::sin(nu);
        const Vector<double, 2> rotated_velocity = {
            cos_nu*velocity[0] - sin_nu*velocity[1],
            sin_nu*velocity[0] + cos_nu*velocity[1]
        };

        const double axial_tilt = obliquity(time);
        const double cos_at = std::cos(axial_tilt);
        const double sin_at = std::sin(axial_tilt);
        return {
            rotated_velocity[0],
            cos_at*rotated_velocity[1],
            sin_at*rotated_velocity[1]
        };
    }

    [[nodiscard]] constexpr
    Vector<double, 3> central_body_direction(double time) const noexcept
    {
        const double ta_nu = true_anomaly(time) + ecliptic_longitude(time);
        const double srev = sin(ta_nu);
        const double axial_tilt = obliquity(time);
        return {
            std::cos(ta_nu), std::cos(axial_tilt)*srev, std::sin(axial_tilt)*srev
        };

    }

    [[nodiscard]] constexpr double eccentric_anomaly(double time) const noexcept
    {
        return kepler_solve<7>(
            m_epoch_mean_speed*time + m_epoch_mean_anomaly, m_epoch_eccentricity);
    }

    [[nodiscard]] constexpr double true_anomaly(double time) const noexcept
    {
        const double ea = eccentric_anomaly(time);
        const double cos_ea = std::cos(ea);
        const double sin_ea = std::sin(ea);

        return ta_from_ea(
            cos_ea, sin_ea, m_epoch_eccentricity, m_axis_ratio);
    }

    [[nodiscard]] constexpr double obliquity(double time) const noexcept
    {
        return m_obliquity(time*(1.0/36525.0));
    }

    [[nodiscard]] constexpr
    double ecliptic_longitude(double time) const noexcept
    {
        return m_ecliptic_longitude(time*(1.0/36525.0));
    }

private:
    Epoch m_epoch;
    double m_epoch_semimajor_axis;
    double m_epoch_eccentricity;
    double m_epoch_mean_anomaly;
    double m_epoch_mean_speed;
    double m_axis_ratio;
    double m_speed_param;
    Polynomial<double, 3> m_obliquity;
    Polynomial<double, 3> m_ecliptic_longitude;
};

template <std::size_t NumIter = 7>
[[nodiscard]] constexpr double kepler_solve(double M, double ecc) noexcept
{
    double E = (ecc > 0.8) ? std::numbers::pi : M;
    for (int i = 0; i < NumIter; i++)
        E -= (E - ecc*std::sin(E) - M)/(1.0 - ecc*std::cos(E));
    return E;
}

[[nodiscard]] constexpr double ta_from_ea(
    double cos_ea, double sin_ea, double ecc, double axis_ratio) noexcept
{
    return std::atan2(
            axis_ratio*sin_ea/(1.0 - ecc*cos_ea),
            (cos_ea - ecc)/(1 - ecc*cos_ea));
}

constexpr Orbit earth_orbit()
{
    return Orbit(
        J2000_epoch(),
        1.49598022961e+8,
        0.01671022,
        -0.0433337328,
        2.0*std::numbers::pi/365.256363004,
        Polynomial<double, 3>({0.40909260, -2.227088e-4, 2.9e-9, 8.790e-9}),
        Polynomial<double, 3>({-1.344825, 2.43802956e-2, -5.38691e-6, -2.9e-11})
    );
}

constexpr GalacticParameters milkyway_coordinates()
{
    return GalacticParameters(
        J2000_epoch(),
        192.729*std::numbers::pi/180.0,
        27.084*std::numbers::pi/180.0,
        122.928*std::numbers::pi/180.0
    );
}

}