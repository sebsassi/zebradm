#pragma once

#include <cmath>
#include "../linalg.hpp"

namespace coordinates
{

/**
    @brief Convert Cartesian to spherical coordinates in geography convention.

    @param cartesian vector of Cartesian coordinates

    @return spherical harmonic coordinates using geography convention in order [azimuth, altitude, length]
*/
[[nodiscard]] constexpr Vector<double, 3> cartesian_to_spherical_geo(
    const Vector<double, 3>& cartesian) noexcept
{
    const double length = std::sqrt(
            cartesian[0]*cartesian[0]
            + cartesian[1]*cartesian[1]
            + cartesian[2]*cartesian[2]);
    const double altitude = std::asin(cartesian[2]/length);
    const double azimuth = std::atan2(cartesian[1], cartesian[0]);

    return {azimuth, altitude, length};
}

/**
    @brief Convert spherical to Cartesian coordinates in geography convention.

    @param spherical vector of spherical coordinates using geography in order [azimuth, altitude, length]

    @return vector of Cartesian coordinates
*/
[[nodiscard]] constexpr Vector<double, 3> spherical_to_cartesian_geo(
    const Vector<double, 3>& spherical) noexcept
{
    const auto& [azimuth, altitude, length] = spherical;
    const Vector<double, 2> vert_rot = {std::cos(altitude), std::sin(altitude)};
    return {
        length*vert_rot[0]*std::cos(azimuth), length*vert_rot[0]*std::sin(azimuth), length*vert_rot[1]
    };
}

/**
    @brief Convert Cartesian to spherical coordinates in physics convention.

    @param cartesian vector of Cartesian coordinates

    @return spherical harmonic coordinates using physics convention in order [azimuth, polar angle, length]
*/
[[nodiscard]] constexpr Vector<double, 3> cartesian_to_spherical_phys(
    const Vector<double, 3>& cartesian) noexcept
{
    const double length = std::sqrt(
            cartesian[0]*cartesian[0]
            + cartesian[1]*cartesian[1]
            + cartesian[2]*cartesian[2]);
    const double polar_angle = std::acos(cartesian[2]/length);
    const double azimuth = std::atan2(cartesian[1], cartesian[0]);

    return {azimuth, polar_angle, length};
}

/**
    @brief Convert spherical to Cartesian coordinates in physics convention.

    @param spherical vector of spherical coordinates using physics in order [azimuth, polar angle, length]

    @return vector of Cartesian coordinates
*/
[[nodiscard]] constexpr Vector<double, 3> spherical_to_cartesian_phys(
    const Vector<double, 3>& spherical) noexcept
{
    const auto& [azimuth, polar_angle, length] = spherical;
    const Vector<double, 2> vert_rot = {std::sin(polar_angle), std::cos(polar_angle)};
    return {
        length*vert_rot[0]*std::cos(azimuth), length*vert_rot[0]*std::sin(azimuth), length*vert_rot[1]
    };
}

/**
    @brief Convert spherical angles to Cartesian coordinates in physics convention.

    @param azimuth azimuthal angle
    @param polar_angle polar angle

    @return vector of Cartesian coordinates
*/
[[nodiscard]] constexpr Vector<double, 3> spherical_to_cartesian_phys(
    double azimuth, double polar_angle) noexcept
{
    const Vector<double, 2> vert_rot = {std::sin(polar_angle), std::cos(polar_angle)};
    return {vert_rot[0]*std::cos(azimuth), vert_rot[0]*std::sin(azimuth), vert_rot[1]};
}

}