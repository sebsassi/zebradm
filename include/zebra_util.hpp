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
#include <span>

#include <zest/real_sh_expansion.hpp>
#include <zest/sh_glq_transformer.hpp>

#include "types.hpp"

namespace zdm
{

/**
    @brief Container that packs multiple spherical harmonic expansions of the same order into one
    buffer.
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
         return View(
            m_data.data(), m_size, m_subspan_size, m_order, m_extent);
    }

    [[nodiscard]] operator ConstView() const noexcept
    {
        return ConstView(
            m_data.data(), m_size, m_subspan_size, m_order, m_extent);
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

    SubSpan operator()(index_type i)
    {
        return SubSpan(m_data.data() + i*m_subspan_size, m_order);
    }

    auto operator[](index_type i)
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

namespace zebra
{

/**
    @brief Class for spherical harmonic expansion of response functions on multiple shells.
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
    template <typename RespType>
    void transform(
        RespType&& resp, std::span<const double> shells, SHExpansionVectorSpan<std::array<double, 2>> out)
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
    template <typename RespType>
    SHExpansionVector transform(RespType&& resp, std::span<const double> shells, std::size_t order)
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
