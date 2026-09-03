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

namespace detail
{

[[nodiscard]] consteval std::size_t count_of(MomentCategory category) noexcept
{
    constexpr std::array<std::size_t, 3> counts = {1, 5, 11};
    return counts[std::to_underlying(category)];
}

[[nodiscard]] consteval std::size_t offset_of(Moment moment) noexcept
{
    constexpr std::array<std::size_t, 11> offsets = {2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
    return offsets[std::to_underlying(moment)];
}

} // namespace detail

template <DistType dist_type, MomentCategory category_param>
class RadonMomentArray
{
public:
    static constexpr MomentCategory category = category_param;

private:
    using expansion_collection_type = std::conditional_t<dist_type == DistType::iso,
        zest::zt::IsotropicZernikeExpansionTensorNormalGeo<
            double, detail::count_of(category)
        >,
        zest::zt::ZernikeExpansionTensorNormalGeo<
            double, zest::IndexingMode::zero_based, detail::count_of(category)
        >
    >;

public:
    using radon_span_type = typename expansion_collection_type::template subspan<1>;
    using const_radon_span_type = typename radon_span_type::const_view;

    RadonMomentArray() = default;

    template <Moment moment>
    [[nodiscard]] radon_span_type access() noexcept
    {
        auto moment_data = m_expansions[std::to_underlying(moment)];
        return radon_span_type{moment_data.data(), moment_data.order() - detail::offset_of(moment)};
    }

    template <Moment moment>
    [[nodiscard]] const_radon_span_type access() noexcept
    {
        auto moment_data = m_expansions[std::to_underlying(moment)];
        return const_radon_span_type{moment_data.data(), moment_data.order() - detail::offset_of(moment)};
    }

private:

    expansion_collection_type m_expansions;
};

} // namespace zdm::zebra
