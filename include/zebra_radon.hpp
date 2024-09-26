#include "zest/zernike_expansion.hpp"
#include "zest/real_sh_expansion.hpp"
#include "zest/sh_glq_transformer.hpp"

namespace zebra
{

/**
    @brief Apply the Radon transform onto a Zernike expansion.

    @param in input Zernike expansion `f_{nlm}`
    @param out output Zernike expansion `g_{nlm}`

    @note The radon transform coefficients are given by `g_{nlm} = f_{nlm}/(2*n + 3) - f_{n - 2,lm}/(2*n - 1)`. The Radon transform in coordinate space is given by `g_{nlm}P_n(w + dot(n,v))Y_{lm}(n)`, where `n` represents a unit vector and `v` is the boost vector.
*/
void radon_transform(
    zest::zt::ZernikeExpansionSpanOrthoGeo<const std::array<double, 2>> in,
    zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>> out) noexcept;

/**
    @brief Apply the Radon transform onto a Zernike expansion.

    @param exp Zernike expansion

    @note This function expects `f_{nlm}` to be contained in the lower portion of `exp`, such that its order is `exp.order() - 2`, and the rest of the values to be zero.

    @note The radon transform coefficients are given by `g_{nlm} = f_{nlm}/(2*n + 3) - f_{n - 2,lm}/(2*n - 1)`. The Radon transform in coordinate space is given by `g_{nlm}P_n(w + dot(n,v))Y_{lm}(n)`, where `n` represents a unit vector and `v` is the boost vector.
*/
void radon_transform_inplace(
    zest::zt::ZernikeExpansionSpanOrthoGeo<std::array<double, 2>> exp) noexcept;

}