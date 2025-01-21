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

namespace zdm
{
namespace zebra
{

template <typename ElementType>
using SHExpansionVectorSpan = SuperSpan<SHExpansionSpan<ElementType>>;

class ResponseTransformer
{
public:
    ResponseTransformer() = default;
    explicit ResponseTransformer(std::size_t order): m_transformer(order) {}

    template <typename RespType>
    void transform(
        RespType&& resp, std::span<const double> min_speeds, SHExpansionVectorSpan<std::array<double, 2>> out)
    {
        for (std::size_t i = 0; i < min_speeds.size(); ++i)
        {
            const double min_speed = min_speeds[i];
            auto surface_func = [&](double lon, double colat) -> double
            {
                return resp(min_speed, lon, colat);
            };
            m_transformer.transform(surface_func, out[i]);
        }
    }
private:
    zest::st::SHTransformerGeo<> m_transformer;
};

} // namespace zebra
} // namespace zdm