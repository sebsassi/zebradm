/*
Copyright (c) 2025-2026 Sebastian Sassi

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

#include <zest/md_array.hpp>
#include <zest/md_span.hpp>

#include "polynomial.hpp"
#include "atomic.hpp"
#include "vector.hpp"

namespace zdm::neft
{

/**
    @brief Squared harmonic oscillator parameter for a given atomic mass number.

    @param mass_number Atomic mass number of an isotope.

    This function calculates the square of the harmonic oscillator parameter
    \f$b\f$ in units of GeV. This function uses the formula given by [1],
    \f[
        b^2 \approx \frac{41.467}{45A^{-1/3} - 25A^{-2/3}}\text{fm}^2.
    \f]

    [1] A. Liam Fitzpatrick *et al.* (2013), JCAP 2013.02, 004,
    [arXiv:1203.3542](https://arxiv.org/abs/1203.3542)
*/
[[nodiscard]] inline double
harmonic_oscillator_parameter_sq(std::size_t mass_number) noexcept
{
    constexpr double fm_as_inv_GeV = 5.06773071615646;
    constexpr double a = 41.467*fm_as_inv_GeV*fm_as_inv_GeV;
    constexpr double b1 = 45.0;
    constexpr double b2 = 25.0;
    const double cbrt_a = std::cbrt(double(mass_number));
    return cbrt_a*cbrt_a*a/(b1*cbrt_a - b2);
}

/**
    @brief Enumerator for specifying isospin basis.
*/
enum class IsospinBasis { nucleon, isospin };

enum class ParticleSpin
{
    scalar = 0,
    fermion = 1,
    vector = 2
};

[[nodiscard]] static constexpr double to_value(ParticleSpin spin_type) noexcept
{
    return double(std::to_underlying(spin_type));
}

[[nodiscard]] static constexpr double spin_factor(ParticleSpin spin_type) noexcept
{
    const double spin = to_value(spin_type);
    return 0.75*spin*(spin + 1.0);
}

/**
    @brief Table for storing nuclear response form factor data.

    @tparam basis_param Choice of isospin basis for the form factors.
    @tparam order Order of the polynomial part of the form factor.

    This structure is used for storing the polynomial parts of the nuclear
    response form factors
    \f[
        F_J^{(N,N')}(y) = e^{-y}P_J^{(N,N')}(y),
    \f]
    where \f$P_J^{(N,N')}(y)\f$ is a polynomial of \f$y = (qb/2)\f$, with
    \f$q\f$ the magnitude of the momentum transfer, and \f$b\f$ the harmonic
    oscillator parameter. The indices \f$N\f$ and \f$N'\f$ are isospin indices,
    with values in \f$\{0,1\}\f$ or in \f$\{n,p\}\f$, depending on whether the
    form factors are expressed in the isospin or nucleon basis. The index
    \f$J\f$ is one of \f$\{M, \Delta, \Sigma', \Sigma'', Delta\Sigma', \Phi'',
    M\Phi''\}\f$ for the different response combinations applicable to the
    effective theory.
*/
template <IsospinBasis basis_param, std::size_t order>
struct NuclearResponseFormFactors
{
    static constexpr IsospinBasis basis = basis_param;
    zest::MDArray<Polynomial<double, order>, 2, 2> m;
    zest::MDArray<Polynomial<double, order>, 2, 2> delta;
    zest::MDArray<Polynomial<double, order>, 2, 2> sigma1;
    zest::MDArray<Polynomial<double, order>, 2, 2> sigma2;
    zest::MDArray<Polynomial<double, order>, 2, 2> delta_sigma1;
    zest::MDArray<Polynomial<double, order>, 2, 2> phi1;
    zest::MDArray<Polynomial<double, order>, 2, 2> phi2;
    zest::MDArray<Polynomial<double, order>, 2, 2> m_phi2;
};

/**
    @brief Class for nonrelativistic nuclear effective theory form factors.

    @tparam order Polynomial order in squared momentum transfer.
    @tparam I Effective theory operator index.
    @tparam J Effective theory operator index.

    This class encodes the dependence of the form factors
    \f$F_{IJ}^{(N,N')}(q^2)\f$ of the nonerlativistic effective theory of DM
    nuclear interactions on the nuclear response form factors. These form
    factors are given by [1]
    \f{align}{
        F_{1,1}^{(N,N')}
            &= F_{M}^{(N,N')},\\
        F_{3,3}^{(N,N')}
            &= \left(
                \frac{q^4}{4 m_N^2} F_{\Phi''}^{(N,N')}
                + q^2v_\perp^2F_{\Sigma'}^{(N,N')}
            \right),\\
        F_{4,4}^{(N,N')}
            &= C(j_\chi)\frac{1}{16}\left(
                F_{\Sigma''}^{(N,N')} +  F_{\Sigma'}^{(N,N')}
            \right),\\
        F_{5,5}^{(N,N')}
            &= C(j_\chi)\frac{1}{4}\left(
                q^2v_\perp^2F_{M}^{(N,N')} + \frac{q^4}{m_N^2}F_{\Delta}^{(N,N')}
            \right),\\
        F_{6,6}^{(N,N')}
            &= C(j_\chi)\frac{q^4}{16}F_{\Sigma''}^{(N,N')},\\
        F_{7,7}^{(N,N')}
            &= \frac{1}{8}v_\perp^2F_{\Sigma'}^{(N,N')},\\
        F_{8,8}^{(N,N')}
            &= C(j_\chi)\frac{1}{4}\left(
                v_\perp^2F_{M}^{(N,N')} + \frac{q^2}{m_N^2}F_{\Delta}^{(N,N')}
            \right),\\
        F_{9,9}^{(N,N')}
            &= C(j_\chi)\frac{q^2}{16}F_{\Sigma'}^{(N,N')},\\
        F_{10,10}^{(N,N')}
            &= \frac{q^2}{4}F_{\Sigma''}^{(N,N')},\\
        F_{11,11}^{(N,N')}
            &= C(j_\chi)\frac{q^2}{4}F_{M}^{(N,N')},\\
        F_{1,3}^{(N,N')}
            &= \frac{q^2}{2m_N}F_{M, \Phi''}^{(N,N')},\\
        F_{4,5}^{(N,N')}
            &= -C(j_\chi)\frac{q^2}{8m_N}F_{\Sigma', \Delta}^{(N,N')},\\
        F_{4,6}^{(N,N')}
            &=  C(j_\chi)\frac{q^2}{16}F_{\Sigma''}^{(N,N')},\\
        F_{8,9}^{(N,N')}
            &= C(j_\chi)\frac{q^2}{8m_N}F_{\Sigma', \Delta}^{(N,N')},
    \f}
    where \f$C(j_\chi) = 4j_\chi(j_\chi + 1)/3\f$ with \f$j_\chi\f$ the DM
    spin, and \f$\vec{v}_\perp\f$ is the transverse velocity.

    [1] A. Liam Fitzpatrick *et al.* (2013), JCAP 2013.02, 004,
    [arXiv:1203.3542](https://arxiv.org/abs/1203.3542)
*/
template <IsospinBasis basis_param, std::size_t order, std::size_t I, std::size_t J>
class EFTFormFactor
{
public:
    static constexpr IsospinBasis basis = basis_param;
    static constexpr std::array<std::size_t, 2> indices = {I, J};

    template <IsospinBasis basis>
    constexpr EFTFormFactor(
        [[maybe_unused]] ParticleSpin spin, [[maybe_unused]] double nucleus_mass,
        [[maybe_unused]] const NuclearResponseFormFactors<basis, order>& nuclear_form_factors) noexcept:
        m_polynomial{} {}

private:
    std::array<Polynomial<double, order>, 2> m_polynomial;
};

template <IsospinBasis basis_param, std::size_t order>
class EFTFormFactor<basis_param, order, 1, 1>
{
public:
    static constexpr IsospinBasis basis = basis_param;
    static constexpr std::array<std::size_t, 2> indices = {1, 1};

    constexpr EFTFormFactor(
        ParticleSpin spin, double nucleus_mass,
        const NuclearResponseFormFactors<basis, order>& nuclear_form_factors) noexcept:
        m_polynomial{init(spin, nucleus_mass, nuclear_form_factors)} {}

private:

    [[nodiscard]] static constexpr std::array<Polynomial<double, order>, 2>
    init(
        [[maybe_unused]] ParticleSpin spin, [[maybe_unused]] double nucleus_mass,
        const NuclearResponseFormFactors<basis, order>& nuclear_form_factors) noexcept
    {
        return {
            nuclear_form_factors.m,
            Polynomial<double, order>{}
        };
    }

    std::array<Polynomial<double, order>, 2> m_polynomial;
};

/**
    @brief The spin-averaged squared matrix element of the nonrelativistic
    effective theory.

    @tparam order Order of the polynomial in squared momentum transfer.

    This class stores the spin-averaged squared matrix element
    \f[
        \frac{1}{2j_\chi + 1}\frac{1}{2j_N + 1}\sum_\text{spins}|\mathcal{M}|^2
            = \sum_{i,j = 1}^{12}\sum_{N,N' = 0,1}
                c_i^Nc_j^{N'}F_{ij}^{(N,N')}(v_\perp^2, q^2)
    \f]
    for a given set of Wilson coefficients and form factors.
*/
template <std::size_t order>
class InteractionFactor
{
public:
    InteractionFactor(const double b_sq, const std::array<Polynomial<double, order>, 2>& polynomial):
        m_polynomial{polynomial}, m_b_sq{b_sq} {}

    [[nodiscard]] constexpr std::array<double, 2>
    operator()(double momentum_transfer) const noexcept
    {
        const double q_sq = momentum_transfer*momentum_transfer;
        const double y = 0.25*q_sq*m_b_sq;

        const double decay = exponential(-2.0*y);
        return {decay*m_polynomial[0](y), decay*m_polynomial[1](y)};
    }

private:
    [[nodiscard]] double exponential(double x)
    {
        return std::exp(x);
    }

    std::array<Polynomial<double, order>, 2> m_polynomial;
    double m_b_sq{};
};

/**
    @brief Class for describing combinations of operators in the
    nonrelativistic effective theory.
*/
template <IsospinBasis basis_param, std::size_t order, typename... FormFactorTypes>
class DMInteraction
{
public:
    static constexpr IsospinBasis basis = basis_param;

    DMInteraction(
        ParticleSpin spin, Isotope isotope,
        const NuclearResponseFormFactors<basis, order>& nuclear_form_factors):
        m_form_factors{FormFactorTypes{spin, isotope.mass(), nuclear_form_factors}...},
        m_b_sq{harmonic_oscillator_parameter_sq(isotope.mass_number)} {}

    [[nodiscard]] constexpr InteractionFactor<order>
    operator()(zest::MDSpan<const double, 15, 2> wilson_coefficients) const noexcept
    {
        return std::apply([&](FormFactorTypes... form_factors)
        {
            std::array<Polynomial<double, order>, 2> res{};
            ([&]{
                const auto& [I, J] = FormFactorTypes::indices;
                res[0] += form_factors[0][0, 0]*wilson_coefficients[I - 1, 0]*wilson_coefficients[J - 1, 0]
                    + form_factors[0][0, 1]*wilson_coefficients[I - 1, 0]*wilson_coefficients[J - 1, 1]
                    + form_factors[0][1, 0]*wilson_coefficients[I - 1, 1]*wilson_coefficients[J - 1, 0]
                    + form_factors[0][1, 1]*wilson_coefficients[I - 1, 1]*wilson_coefficients[J - 1, 1];
                res[1] += form_factors[1][0, 0]*wilson_coefficients[I - 1, 0]*wilson_coefficients[J - 1, 0]
                    + form_factors[1][0, 1]*wilson_coefficients[I - 1, 0]*wilson_coefficients[J - 1, 1]
                    + form_factors[1][1, 0]*wilson_coefficients[I - 1, 1]*wilson_coefficients[J - 1, 0]
                    + form_factors[1][1, 1]*wilson_coefficients[I - 1, 1]*wilson_coefficients[J - 1, 1];
            }(), ...);
            return InteractionFactor{m_b_sq, res};
        }, m_form_factors);
    }

private:
    std::tuple<FormFactorTypes...> m_form_factors;
    double m_b_sq;
};

} // namespace zdm::neft
