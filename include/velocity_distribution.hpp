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

#include <zest/zernike_glq_transformer.hpp>

#include "matrix.hpp"
#include "vector.hpp"

namespace zdm
{

template <typename FieldType>
concept bounded_distribution = requires (const FieldType& dist, const la::Vector<double, 3>& velocity)
{
    { dist(velocity) } -> std::same_as<double>;
    { dist.normalization() } -> std::same_as<double>;
    { dist.max_velocity() } -> std::same_as<double>;
};

template <bounded_distribution Func>
zest::zt::RealZernikeExpansionNormalGeo zernike_transform(const Func& dist, std::size_t lmax, const la::RotationMatrix<double, 3>& rotation)
{
    const double scale = dist.max_velocity();
    auto dist_wrap = [&](const std::array<double, 3>& x)
    {
        return dist(rotation*(scale*la::Vector(x)));
    };

    zest::zt::BallGLQGridPoints points(lmax);
    return zest::zt::GLQTransformerNormalGeo(lmax).forward_transform(
            points.generate_values(dist_wrap, lmax), lmax);
}

} // namespace zdm
