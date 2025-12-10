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

#include <algorithm>
#include <array>
#include <concepts>
#include <span>

#include <zest/radial_zernike_recursion.hpp>
#include <zest/real_sh_expansion.hpp>
#include <zest/sh_glq_transformer.hpp>
#include <zest/zernike_expansion.hpp>
#include <zest/sh_generator.hpp>

#include "vector.hpp"
#include "coordinate_transforms.hpp"
#include "types.hpp"

namespace zdm
{

/**
    @brief Container that packs multiple spherical harmonic expansions of the
    same order into one buffer.
*/
class SHExpansionVector
{
public:
    using element_type = std::array<double, 2>;
    using value_type = element_type;
    using size_type = std::size_t;
    using index_type = std::size_t;
    using View = SHExpansionVectorSpan<element_type>;
    using ConstView = SHExpansionVectorSpan<const element_type>;
    using SubSpan = SHExpansionSpan<element_type>;
    using ConstSubSpan = SHExpansionSpan<const element_type>;

    [[nodiscard]] static constexpr size_type size(size_type extent, size_type order)
    {
        return extent*SubSpan::size(order);
    }

    SHExpansionVector() = default;

    SHExpansionVector(size_type extent, size_type order) noexcept:
        m_data(size(extent, order)), m_subspan_size(SubSpan::size(order)), m_order(order), m_extent(extent) {}
   
    [[nodiscard]] operator View() noexcept
    {
         return View(m_data.data(), m_size, m_subspan_size, m_order, m_extent);
    }

    [[nodiscard]] operator ConstView() const noexcept
    {
        return ConstView(m_data.data(), m_size, m_subspan_size, m_order, m_extent);
    }
    
    [[nodiscard]] size_type size() const noexcept
    {
        return m_data.size();
    }
    
    [[nodiscard]] size_type subspan_size() const noexcept
    {
        return m_subspan_size;
    }

    [[nodiscard]] size_type extent() const noexcept
    {
        return m_extent;
    }

    [[nodiscard]] std::span<element_type> flatten() noexcept
    {
        return std::span<element_type>(m_data);
    }

    [[nodiscard]] std::span<const element_type> flatten() const noexcept
    {
        return std::span<const element_type>(m_data);
    }

    [[nodiscard]] SubSpan operator()(index_type i) noexcept
    {
        assert(i < m_size);
        return SubSpan(m_data.data() + i*m_subspan_size, m_order);
    }

    [[nodiscard]] ConstSubSpan operator()(index_type i) const noexcept
    {
        assert(i < m_size);
        return ConstSubSpan(m_data.data() + i*m_subspan_size, m_order);
    }

    [[nodiscard]] SubSpan operator[](index_type i) noexcept
    {
        return (*this)(i);
    }

    [[nodiscard]] ConstSubSpan operator[](index_type i) const noexcept
    {
        return (*this)(i);
    }

private:
    std::vector<std::array<double, 2>> m_data;
    size_type m_size = {};
    size_type m_subspan_size = {};
    size_type m_order = {};
    size_type m_extent = {};
};

[[nodiscard]] ZernikeExpansion
from_points(std::span<la::Vector<double, 3>> points, std::span<double> values, std::size_t order)
{
    if (points.size() != values.size())
        throw std::runtime_error("Number of values differs from number of points");

    std::vector<double> radii(points.size());
    std::vector<double> colatitudes(points.size());
    std::vector<double> longitudes(points.size());

    for (std::size_t i = 0; i < points.size(); ++i)
    {
        const auto& [longitude, colatitude, radial] = coordinates::cartesian_to_spherical_phys(points[i]);
        radii[i] = radial;
        colatitudes[i] = colatitude;
        longitudes[i] = longitude;
    }

    const double max_radius = *std::ranges::max_element(radii);

    for (auto& radius : radii)
        radius *= 1.0/max_radius;

    std::vector<double> radial_zernike_buffer(zest::zt::RadialZernikeVecSpan<double, ZernikeExpansion::zernike_norm>::size(order));
    zest::zt::RadialZernikeVecSpan<double, ZernikeExpansion::zernike_norm>
    radial_zernike(radial_zernike_buffer.data(), order);

    zest::zt::RadialZernikeRecursion(order).zernike(radii, radial_zernike);

    zest::st::RealSHExpansionVector<ZernikeExpansion::sh_norm, ZernikeExpansion::sh_phase, double>
    spherical_harmonics(order);

    zest::st::RealSHGenerator().generate(longitudes, colatitudes, spherical_harmonics);

    ZernikeExpansion expansion(order);

    std::vector<double> partial_integrand(points.size());

    const double prefactor = (4.0*std::numbers::pi/(3.0*double(points.size())));
    for (std::size_t n : expansion.indices())
    {
        auto radial_zernike_n = radial_zernike[n];
        auto expansion_n = expansion[n];
        for (std::size_t l : expansion_n.indices())
        {
            auto radial_zernike_nl = radial_zernike_n[l];
            for (std::size_t i = 0; i < points.size(); ++i)
                partial_integrand = values[i]*radial_zernike_nl[i];

            auto spherical_harmonics_l = spherical_harmonics[l];
            auto expansion_nl = expansion_n[l];
            for (std::size_t m : expansion_nl.indices())
            {
                auto spherical_harmonics_lm = spherical_harmonics_l[m];
                std::array<double, 2>& expansion_nlm = expansion_nl[m];
                for (std::size_t i = 0; i < points.size(); ++i)
                {
                    expansion_nlm[0] += partial_integrand[i]*spherical_harmonics_lm[0,i];
                    expansion_nlm[1] += partial_integrand[i]*spherical_harmonics_lm[1,i];
                }
                expansion_nlm[0] *= prefactor;
                expansion_nlm[1] *= prefactor;
            }
        }
    }

    return expansion;
}

[[nodiscard]] ZernikeExpansion
from_triangulation(
    std::span<la::Vector<double, 3>> points, std::span<std::array<std::size_t, 4>> simplices,
    std::span<double> values, std::size_t order)
{

}

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
    void transform(
        const RespType& resp, std::span<const double> shells, SHExpansionVectorSpan<std::array<double, 2>> out)
    {
        for (std::size_t i = 0; i < shells.size(); ++i)
        {
            const double shell = shells[i];
            auto surface_func = [&](double lon, double colat) -> double
            {
                return resp(shell, lon, colat);
            };
            m_transformer.transform(surface_func, out[i]);
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
    SHExpansionVector transform(const RespType& resp, std::span<const double> shells, std::size_t order)
    {
        SHExpansionVector res(shells.size(), order);
        transform(resp, shells, res);
        return res;
    }
private:
    zest::st::SHTransformerGeo<> m_transformer;
};

} // namespace zebra
} // namespace zdm
