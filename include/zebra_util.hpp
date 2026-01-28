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

#include <cassert>
#include <concepts>
#include <span>

#include <zest/radial_zernike_recursion.hpp>
#include <zest/sequence.hpp>
#include <zest/sh_conventions.hpp>
#include <zest/sh_expansion.hpp>
#include <zest/sh_generator.hpp>
#include <zest/sh_glq_transformer.hpp>
#include <zest/zernike_expansion.hpp>

#include "vector.hpp"
#include "coordinate_transforms.hpp"
#include "types.hpp"

namespace zdm
{

[[nodiscard]] ZernikeExpansion
from_points(std::span<la::Vector<double, 3>> points, std::span<double> values, std::size_t order)
{
    assert(points.size() == values.size());

    std::vector<double> radii(points.size());
    std::vector<double> colatitudes(points.size());
    std::vector<double> longitudes(points.size());

    for (std::size_t i = 0; i < points.size(); ++i)
    {
        const auto& [longitude, colatitude, radial]
            = coordinates::cartesian_to_spherical_phys(points[i]);
        radii[i] = radial;
        colatitudes[i] = colatitude;
        longitudes[i] = longitude;
    }

    const double max_radius = *std::ranges::max_element(radii);

    for (auto& radius : radii)
        radius *= 1.0/max_radius;

    constexpr zest::IndexingMode indexing_mode = zest::IndexingMode::zero_based;
    constexpr zest::zt::ZernikeNorm zernike_norm = zest::zt::zernike_norm_of<ZernikeExpansion>();
    constexpr zest::st::SHNorm sh_norm = zest::st::sh_norm_of<ZernikeExpansion>();
    constexpr zest::st::SHPhase sh_phase = zest::st::sh_phase_of<ZernikeExpansion>();

    zest::zt::RadialZernikeExpansion<double, zernike_norm, std::dynamic_extent>
    radial_zernike{order, points.size()};

    zest::zt::RadialZernikeRecursion{order}.generate(radii, radial_zernike);

    auto spherical_harmonics = zest::st::RealSHGenerator()
        .generate<indexing_mode, sh_norm, sh_phase>(longitudes, colatitudes, order);

    ZernikeExpansion expansion{order};

    std::vector<double> partial_integrand(points.size());

    const double prefactor = 1.0/(3.0*double(points.size()));
    for (std::size_t n : expansion.indices())
    {
        auto radial_zernike_n = radial_zernike[n];
        auto expansion_n = expansion[n];
        for (std::size_t l : expansion_n.indices())
        {
            auto radial_zernike_nl = radial_zernike_n[l];
            for (std::size_t i = 0; i < points.size(); ++i)
                partial_integrand[i] = values[i]*radial_zernike_nl[i];

            auto spherical_harmonics_l = spherical_harmonics[l];
            auto expansion_nl = expansion_n[l];
            for (std::size_t m : expansion_nl.indices())
            {
                auto spherical_harmonics_lm = spherical_harmonics_l[m];
                auto expansion_nlm = expansion_nl[m];
                for (std::size_t i = 0; i < points.size(); ++i)
                {
                    expansion_nlm[0] += partial_integrand[i]*spherical_harmonics_lm[0, i];
                    expansion_nlm[1] += partial_integrand[i]*spherical_harmonics_lm[1, i];
                }
                expansion_nlm[0] *= prefactor;
                expansion_nlm[1] *= prefactor;
            }
        }
    }

    return expansion;
}

// [[nodiscard]] ZernikeExpansion
// from_triangulation(
//     std::span<la::Vector<double, 3>> points, std::span<std::array<std::size_t, 4>> simplices,
//     std::span<double> values, std::size_t order)
// {
//
// }

namespace zebra
{

/**
    @brief Transforms functions into their spherical harmonic expansions.
*/
class ResponseTransformer
{
public:
    ResponseTransformer() = default;
    explicit ResponseTransformer(std::size_t order): m_transformer(order) {}

    /**
        @brief Take spherical harmonic transform of a response function.

        @tparam RespType type of response function

        @param resp response function
        @param shells shells the spherical harmonic transforms are evaluated on
        @param out spherical harmonic expansions of the response
    */
    template <std::regular_invocable<double, double, double> RespType>
    void forward_transform(
        const RespType& resp, std::span<const double> shells, SHVectorSpan<double> out)
    {
        assert(shells.size() == out.extent(0));
        for (std::size_t i = 0; i < shells.size(); ++i)
        {
            const double shell = shells[i];
            auto surface_func = [&](double lon, double colat) -> double
            {
                return resp(shell, lon, colat);
            };
            m_transformer.forward_transform(surface_func, out[i]);
        }
    }

    /**
        @brief Take spherical harmonic transform of a response function.

        @tparam RespType type of response function

        @param resp response function
        @param shells shells the spherical harmonic transforms are evaluated on
        @param order order of spherical harmonic expansions

        @return spherical harmonic expansions of the response
    */
    template <std::regular_invocable<double, double, double> RespType>
    SHExpansionVector forward_transform(const RespType& resp, std::span<const double> shells, std::size_t order)
    {
        SHExpansionVector res(shells.size(), order);
        forward_transform(resp, shells, res);
        return res;
    }
private:
    zest::st::SHTransformerGeo<> m_transformer;
};

} // namespace zebra

} // namespace zdm

