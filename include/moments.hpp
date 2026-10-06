/*
Copyright (c) 2024-2026 Sebastian Sassi

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

#include <cstdint>

#include "types.hpp"

namespace zdm
{

enum class MomentSet: std::uint8_t
{
    identity,
    transverse,
    full,
};

enum class Moment: std::uint8_t
{
    identity,
    x,
    y,
    z,
    r2,
    x2,
    y2,
    z2,
    xy,
    xz,
    yz,
};

enum class IsoMoment: std::uint8_t
{
    identity,
    linear,
    quadratic,
};

[[nodiscard]] constexpr std::string_view to_string(MomentSet category) noexcept
{
    constexpr std::array strings = {
        "identity",
        "transverse",
        "full",
    };

    return strings[std::to_underlying(category)];
}

[[nodiscard]] constexpr std::string_view to_string(Moment moment) noexcept
{
    constexpr std::array strings = {
        "identity",
        "x",
        "y",
        "z",
        "r2",
        "x2",
        "y2",
        "z2",
        "xy",
        "xz",
        "yz",
    };

    return strings[std::to_underlying(moment)];
}

[[nodiscard]] constexpr std::string_view to_string(IsoMoment moment) noexcept
{
    constexpr std::array strings = {
        "identity",
        "linear",
        "quadratic",
    };

    return strings[std::to_underlying(moment)];
}

namespace detail
{

[[nodiscard]] constexpr std::size_t
moment_count(DistType dist_type, MomentSet category) noexcept
{
    if (dist_type == DistType::iso)
        return (category == MomentSet::identity) ? 1 : 3;
    else
    {
        constexpr std::array<std::size_t, 3> counts = {1, 5, 11};
        return counts[std::to_underlying(category)];
    }
}

[[nodiscard]] constexpr std::size_t
max_offset(MomentSet category) noexcept
{
    return (category == MomentSet::identity) ? 0 : 2;
}

[[nodiscard]] constexpr std::size_t
offset_of(Moment moment) noexcept
{
    constexpr std::array<std::size_t, 11> offsets = {0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2};
    return offsets[std::to_underlying(moment)];
}

[[nodiscard]] constexpr std::size_t
inverse_offset_of(Moment moment) noexcept
{
    constexpr std::array<std::size_t, 11> offsets = {2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
    return offsets[std::to_underlying(moment)];

}

[[nodiscard]] constexpr std::size_t
offset_of(IsoMoment moment) noexcept
{
    return (moment != IsoMoment::quadratic) ? 0 : 2;
}

[[nodiscard]] constexpr std::size_t
inverse_offset_of(IsoMoment moment) noexcept
{
    return (moment == IsoMoment::quadratic) ? 0 : 2;
}

} // namespace detail

[[nodiscard]] constexpr std::size_t
max_radon_order(MomentSet category, std::size_t zernike_order)
{
    return zernike_order + 2 + detail::max_offset(category);
}

} // namespace zdm
