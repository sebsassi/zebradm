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

#include <concepts>
#include <span>

#include <zest/sh_expansion.hpp>
#include <zest/sh_glq_transformer.hpp>

#include "types.hpp"



namespace zdm::zebra
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
        const RespType& resp, std::span<const double> shells, SHVectorSpan<double> out)
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

} // namespace zdm::zebra

