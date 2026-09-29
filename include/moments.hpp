
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

namespace zdm::zebra
{

enum class MomentCategory: std::uint8_t
{
    identity,
    transverse,
    full
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
    yz
};

enum class IsoMoment: std::uint8_t
{
    identity,
    linear,
    quadratic
};

namespace detail
{

[[nodiscard]] consteval std::size_t count_of(DistType dist_type, MomentCategory category) noexcept
{
    if (dist_type == DistType::iso)
        return (category == MomentCategory::identity) ? 1 : 3;
    else
    {
        constexpr std::array<std::size_t, 3> counts = {1, 5, 11};
        return counts[std::to_underlying(category)];
    }
}

[[nodiscard]] consteval std::size_t max_offset([[maybe_unused]] DistType dist_type, MomentCategory category) noexcept
{
    return (category == MomentCategory::identity) ? 0 : 2;
}

[[nodiscard]] consteval std::size_t offset_of(Moment moment) noexcept
{
    constexpr std::array<std::size_t, 11> offsets = {0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2};
    return offsets[std::to_underlying(moment)];
}

[[nodiscard]] consteval std::size_t offset_of(IsoMoment moment) noexcept
{
    constexpr std::array<std::size_t, 3> offsets = {0, 2, 4};
    return offsets[std::to_underlying(moment)];
}

} // namespace detail

} // namespace zdm::zebra
