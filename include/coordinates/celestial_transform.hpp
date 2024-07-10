#pragma once

#include "astro_parameters.hpp"
#include "time.hpp"

namespace coordinates
{

class GalacticToHeliocentric
{
public:
    constexpr GalacticToHeliocentric(const Vector<double, 3>& p_peculiar_velocity, double p_circular_speed) : m_peculiar_velocity(p_peculiar_velocity), m_circular_speed(p_circular_speed) {}

    [[nodiscard]] constexpr
    Matrix<double, 4, 4> transform(double time) const noexcept
    {
        return affine_matrix(rotation(time), boost(time));
    }

    [[nodiscard]] constexpr
    Matrix<double, 3, 3> rotation(double time) const noexcept
    {
        return identity<double, 3>();
    }

    [[nodiscard]] constexpr Vector<double, 3> boost(double time) const noexcept
    {
        return -solar_velocity();
    }

    [[nodiscard]] constexpr Vector<double, 3> solar_velocity() const noexcept
    {
        return {
            m_peculiar_velocity[0], m_peculiar_velocity[1] + m_circular_speed, m_peculiar_velocity[2]
        };
    }

private:
    Vector<double, 3> m_peculiar_velocity;
    double m_circular_speed;
};

class HeliocentricToEquatorial
{
public:
    constexpr HeliocentricToEquatorial(const GalacticParameters& p_galactic_angles, const Orbit& p_orbital_parameters) : m_galactic_angles(p_galactic_angles), m_orbital_parameters(p_orbital_parameters), m_rot_hel_to_equ(p_galactic_angles.rotation_to_equ()) {}

    [[nodiscard]] constexpr
    Matrix<double, 4, 4> transform(double time) const noexcept
    {
        return affine_matrix(rotation(time), boost(time));
    }

    [[nodiscard]] constexpr
    Matrix<double, 3, 3> rotation(double time) const noexcept
    {
        return m_rot_hel_to_equ;
    }

    [[nodiscard]] constexpr Vector<double, 3> boost(double time) const noexcept
    {
        return -m_orbital_parameters.orbital_velocity(time);
    }
private:
    GalacticParameters m_galactic_angles;
    Orbit m_orbital_parameters;
    Matrix<double, 3, 3> m_rot_hel_to_equ;
};

class EquatorialToHorizontal
{
public:
    constexpr EquatorialToHorizontal(double p_longitude, double p_latitude, Epoch p_epoch, double p_epoch_ra, double p_epoch_ra_rate, double p_radius) : m_epoch(p_epoch), m_longitude(p_longitude), m_latitude(p_latitude), m_epoch_ra(p_epoch_ra), m_epoch_ra_rate(p_epoch_ra_rate) {}

    [[nodiscard]] constexpr
    Matrix<double, 4, 4> transform(double time) const noexcept
    {
        return affine_matrix(rotation(time), boost(time));
    }

    [[nodiscard]] constexpr
    Matrix<double, 3, 3> rotation(double time) const noexcept
    {
        const double ra = rotation_angle(time) + m_longitude - 0.5*std::numbers::pi;

        const double clat = std::cos(m_latitude);
        const double slat = std::sin(m_latitude);
        const double crot = std::cos(ra);
        const double srot = std::sin(ra);

        return {
            Vector<double, 3>{crot, srot, 0.0},
            Vector<double, 3>{-slat*srot, slat*crot, -clat},
            Vector<double, 3>{-clat*srot, clat*crot, slat}
        };
    }

    [[nodiscard]] constexpr Vector<double, 3> boost(double time) const noexcept
    {
        const double rotation_speed = m_radius*(2.0*std::numbers::pi/86400.0)*m_epoch_ra_rate*std::sin(m_latitude);
        return {-rotation_speed, 0.0, 0.0};
    }

    [[nodiscard]] constexpr double rotation_angle(double time) const noexcept
    {
        return 2.0*std::numbers::pi*(m_epoch_ra + m_epoch_ra_rate*time);
    }
private:
    Epoch m_epoch;
    double m_longitude;
    double m_latitude;
    double m_epoch_ra;
    double m_epoch_ra_rate;
    double m_radius;
};

}