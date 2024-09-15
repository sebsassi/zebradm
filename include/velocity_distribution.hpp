#pragma once

#include "linalg.hpp"
#include "zest/zernike.hpp"

template <typename FieldType>
concept bounded_distribution = requires (const FieldType& dist, const Vector<double, 3>& velocity)
{
    { dist(velocity) } -> std::same_as<double>;
    { dist.normalization() } -> std::same_as<double>;
    { dist.max_velocity() } -> std::same_as<double>;
};

template <bounded_distribution Func>
zest::zt::ZernikeExpansion zernike_transform(const Func& dist, std::size_t lmax, const Matrix<double, 3, 3>& rotation)
{
    const double scale = dist.max_velocity();
    auto dist_ = [&](const Vector<double, 3>& x)
    { 
        return dist(rotation*(scale*x));
    };

    zest::zt::BallGLQGridPoints points(lmax);
    return zest::zt::GLQTransformer(lmax).transform(
            points.generate_values(dist_, lmax), lmax);
}