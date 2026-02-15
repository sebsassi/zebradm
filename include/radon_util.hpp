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
#include <cassert>

#include <zest/rotor.hpp>
#include <zest/zernike_conventions.hpp>

namespace zdm::util
{

template <zest::zt::ZernikeNorm NORM>
inline double geg_rec_coeff(std::size_t n) noexcept
{
    if constexpr (NORM == zest::zt::ZernikeNorm::unnormed)
        return 1.0/double(2*n + 3);
    else
        return 1.0/std::sqrt(double(2*n + 3));
}

/**
    @brief Set Euler angles for rotation to a coordinate system whose z-axis is in the direction given by the arguments.

    @note The convention for the Euler angles is that used by `zest::Rotor::rotate`.
*/
template <zest::RotationType TYPE>
constexpr std::array<double, 3> euler_angles_to_align_z(
    double azimuth, double colatitude)
{
    assert(0.0 <= colatitude && colatitude <= std::numbers::pi);
    if constexpr (TYPE == zest::RotationType::coordinate)
        return {azimuth, colatitude, 0.0};
    else
        return {0.0, -colatitude, std::numbers::pi - azimuth};
}

} // namespace zdm::util
