/*
Copyright (c) 2024 Sebastian Sassi

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

#include <array>
#include <cmath>

#include "linalg.hpp"

namespace zdm
{

namespace coordinates
{

/**
    @brief Convert Cartesian to spherical coordinates in geography convention.

    @param cartesian vector of Cartesian coordinates

    @return spherical harmonic coordinates in order [longitude, latitude, length]
*/
[[nodiscard]] constexpr std::array<double, 3> cartesian_to_spherical_geo(
    const std::array<double, 3>& cartesian) noexcept
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

    @param spherical vector of spherical coordinates in order [longitude, latitude, length]

    @return vector of Cartesian coordinates
*/
[[nodiscard]] constexpr std::array<double, 3> spherical_to_cartesian_geo(
    const std::array<double, 3>& spherical) noexcept
{
    const auto& [azimuth, altitude, length] = spherical;
    const std::array<double, 2> vert_rot = {std::cos(altitude), std::sin(altitude)};
    return {
        length*vert_rot[0]*std::cos(azimuth), length*vert_rot[0]*std::sin(azimuth), length*vert_rot[1]
    };
}

/**
    @brief Convert spherical angles to Cartesian coordinates in physics convention.

    @param longitude longitude angle in radians [0, 2pi]
    @param latitude latitude in radians [-pi/2, pi/2]

    @return vector of Cartesian coordinates
*/
[[nodiscard]] constexpr std::array<double, 3> spherical_to_cartesian_geo(
    double longitude, double latitude) noexcept
{
    const std::array<double, 2> vert_rot = {std::cos(latitude), std::sin(colatitude)};
    return {vert_rot[0]*std::cos(longitude), vert_rot[0]*std::sin(longitude), vert_rot[1]};
}

/**
    @brief Convert Cartesian to spherical coordinates in physics convention.

    @param cartesian vector of Cartesian coordinates

    @return spherical harmonic coordinates using in order [azimuth, colatitude, length]
*/
[[nodiscard]] constexpr std::array<double, 3> cartesian_to_spherical_phys(
    const std::array<double, 3>& cartesian) noexcept
{
    const double length = std::sqrt(
            cartesian[0]*cartesian[0]
            + cartesian[1]*cartesian[1]
            + cartesian[2]*cartesian[2]);
    const double colatitude = std::acos(cartesian[2]/length);
    const double azimuth = std::atan2(cartesian[1], cartesian[0]);

    return {azimuth, colatitude, length};
}

/**
    @brief Convert spherical to Cartesian coordinates in physics convention.

    @param spherical vector of spherical coordinates in order [azimuth, colatitude, length]

    @return vector of Cartesian coordinates
*/
[[nodiscard]] constexpr std::array<double, 3> spherical_to_cartesian_phys(
    const std::array<double, 3>& spherical) noexcept
{
    const auto& [azimuth, colatitude, length] = spherical;
    const std::array<double, 2> vert_rot = {std::sin(colatitude), std::cos(colatitude)};
    return {
        length*vert_rot[0]*std::cos(azimuth), length*vert_rot[0]*std::sin(azimuth), length*vert_rot[1]
    };
}

/**
    @brief Convert spherical angles to Cartesian coordinates in physics convention.

    @param azimuth azimuthal angle in radians [0, 2pi]
    @param colatitude colatitude in radians [0, pi]

    @return vector of Cartesian coordinates
*/
[[nodiscard]] constexpr std::array<double, 3> spherical_to_cartesian_phys(
    double azimuth, double colatitude) noexcept
{
    const std::array<double, 2> vert_rot = {std::sin(colatitude), std::cos(colatitude)};
    return {vert_rot[0]*std::cos(azimuth), vert_rot[0]*std::sin(azimuth), vert_rot[1]};
}

} // namespace coordinates
} // namespace zdm
