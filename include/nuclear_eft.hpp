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
    @brief Choice of isospin basis for nuclear form factor computations.
*/
enum class IsospinBasis
{
    nucleon,
    isospin
};

/**
    @brief Labels for particle spin representations.
*/
enum class ParticleSpin
{
    scalar = 0,
    fermion = 1,
    vector = 2
};

enum class TransverseVelocityDependence
{
    none,
    mono,
    poly
};

/**
    @brief Convert particle spin representation label to spin value.
*/
[[nodiscard]] static constexpr double to_value(ParticleSpin spin_type) noexcept
{
    return 0.5*double(std::to_underlying(spin_type));
}

/**
    @brief Calculate spin factor in the nonrelativistic nuclear effective
    theory of DM interactions.
*/
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
        F_K^{(N,N')}(y) = e^{-y}P_K^{(N,N')}(y),
    \f]
    where \f$P_K^{(N,N')}(y)\f$ is a polynomial of \f$y = (qb/2)\f$, with
    \f$q\f$ the magnitude of the momentum transfer, and \f$b\f$ the harmonic
    oscillator parameter. The indices \f$N\f$ and \f$N'\f$ are isospin indices,
    with values in \f$\{0,1\}\f$ or in \f$\{n,p\}\f$, depending on whether the
    form factors are expressed in the isospin or nucleon basis. The index
    \f$K\f$ is one of \f$\{M, \Delta, \Sigma', \Sigma'', Delta\Sigma', \Phi'',
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

/*
    This is just a convoluted way of implementing a function-like object to
    make use of partial template specialization. You know, because this is
    C++. Being able to just write a partial specialization for a template
    function would be too straightforward.
*/
namespace detail
{

// This is miscellaneous data that appears in the definition of multiple EFT
// form factors, and is collected here to avoid unnecessary duplication.
struct EFTFormFactorInputData
{
    double eft_spin_factor{};
    double inverse_nucleus_mass{};
    double quad_inverse_b_sq{};

    EFTFormFactorInputData(ParticleSpin spin, Isotope isotope):
        eft_spin_factor(spin_factor(spin)),
        inverse_nucleus_mass{1.0/(isotope.mass()*isotope.mass())},
        quad_inverse_b_sq{4.0/harmonic_oscillator_parameter_sq(isotope.mass_number)} {}
};

template <IsospinBasis basis, std::size_t order, std::size_t I, std::size_t J>
struct EFTFormFactorInitHelper
{
    // This is actually what we care about specializing. All else is just
    // boilerplate.
    static constexpr void operator()(
        [[maybe_unused]] const EFTFormFactorInputData& input_data,
        [[maybe_unused]] const NuclearResponseFormFactors<basis, order>& nuclear_form_factors) noexcept {}
};


template <IsospinBasis basis, std::size_t order>
struct EFTFormFactorInitHelper<basis, order, 1, 1>
{
    [[nodiscard]] static constexpr Polynomial<double, order> operator()(
        [[maybe_unused]] const EFTFormFactorInputData& input_data,
        [[maybe_unused]] const NuclearResponseFormFactors<basis, order>& nuclear_form_factors) noexcept
    {
        return nuclear_form_factors.m;
    }
};

template <IsospinBasis basis, std::size_t order>
struct EFTFormFactorInitHelper<basis, order, 3, 3>
{
    [[nodiscard]] static constexpr std::array<Polynomial<double, order>, 2> operator()(
        const EFTFormFactorInputData& input_data,
        [[maybe_unused]] const NuclearResponseFormFactors<basis, order>& nuclear_form_factors) noexcept
    {
        const double nontrans_coeff
            = 0.25*input_data.inverse_nucleus_mass*input_data.inverse_nucleus_mass
                *input_data.quad_inverse_b_sq*input_data.quad_inverse_b_sq;
        const double trans_coeff
            = 0.125*input_data.quad_inverse_b_sq;
        return {
            Monomial<double, 2>{nontrans_coeff}*nuclear_form_factors.phi2,
            Monomial<double, 1>{trans_coeff}*nuclear_form_factors.sigma1
        };
    }
};

template <IsospinBasis basis, std::size_t order>
struct EFTFormFactorInitHelper<basis, order, 4, 4>
{
    [[nodiscard]] static constexpr Polynomial<double, order> operator()(
        const EFTFormFactorInputData& input_data,
        [[maybe_unused]] const NuclearResponseFormFactors<basis, order>& nuclear_form_factors) noexcept
    {
        return 0.0625*input_data.eft_spin_factor*(nuclear_form_factors.sigma1 + nuclear_form_factors.sigma2);
    }
};

template <IsospinBasis basis, std::size_t order>
struct EFTFormFactorInitHelper<basis, order, 5, 5>
{
    [[nodiscard]] static constexpr std::array<Polynomial<double, order>, 2> operator()(
        const EFTFormFactorInputData& input_data,
        [[maybe_unused]] const NuclearResponseFormFactors<basis, order>& nuclear_form_factors) noexcept
    {
        const double nontrans_coeff
            = 0.25*input_data.eft_spin_factor
                *input_data.inverse_nucleus_mass*input_data.inverse_nucleus_mass
                *input_data.quad_inverse_b_sq*input_data.quad_inverse_b_sq;
        const double trans_coeff 
            = 0.25*input_data.eft_spin_factor*input_data.quad_inverse_b_sq;
        return {
            Monomial<double, 2>{nontrans_coeff}*nuclear_form_factors.delta,
            Monomial<double, 1>{trans_coeff}*nuclear_form_factors.m
        };
    }
};

template <IsospinBasis basis, std::size_t order>
struct EFTFormFactorInitHelper<basis, order, 6, 6>
{
    [[nodiscard]] static constexpr Polynomial<double, order> operator()(
        const EFTFormFactorInputData& input_data,
        [[maybe_unused]] const NuclearResponseFormFactors<basis, order>& nuclear_form_factors) noexcept
    {
        const double coeff
            = 0.0625*input_data.eft_spin_factor
                *input_data.quad_inverse_b_sq*input_data.quad_inverse_b_sq;
        return Monomial<double, 2>{coeff}*nuclear_form_factors.sigma2;
    }
};

template <IsospinBasis basis, std::size_t order>
struct EFTFormFactorInitHelper<basis, order, 7, 7>
{
    [[nodiscard]] static constexpr Polynomial<double, order> operator()(
        [[maybe_unused]] const EFTFormFactorInputData& input_data,
        [[maybe_unused]] const NuclearResponseFormFactors<basis, order>& nuclear_form_factors) noexcept
    {
        return 0.125*nuclear_form_factors.sigma1;
    }
};

template <IsospinBasis basis, std::size_t order>
struct EFTFormFactorInitHelper<basis, order, 8, 8>
{
    [[nodiscard]] static constexpr std::array<Polynomial<double, order>, 2> operator()(
        const EFTFormFactorInputData& input_data,
        [[maybe_unused]] const NuclearResponseFormFactors<basis, order>& nuclear_form_factors) noexcept
    {
        const double nontrans_coeff
            = 0.25*input_data.eft_spin_factor
                *input_data.inverse_nucleus_mass*input_data.inverse_nucleus_mass
                *input_data.quad_inverse_b_sq;
        const double trans_coeff
            = 0.25*input_data.eft_spin_factor;
        return {
            Monomial<double, 2>{nontrans_coeff}*nuclear_form_factors.delta,
            Monomial<double, 1>{trans_coeff}*nuclear_form_factors.m
        };
    }
};

template <IsospinBasis basis, std::size_t order>
struct EFTFormFactorInitHelper<basis, order, 9, 9>
{
    [[nodiscard]] static constexpr Polynomial<double, order> operator()(
        const EFTFormFactorInputData& input_data,
        [[maybe_unused]] const NuclearResponseFormFactors<basis, order>& nuclear_form_factors) noexcept
    {
        const double coeff = 0.0625*input_data.eft_spin_factor*input_data.quad_inverse_b_sq;
        return Monomial<double, 1>{coeff}*nuclear_form_factors.sigma1;
    }
};

template <IsospinBasis basis, std::size_t order>
struct EFTFormFactorInitHelper<basis, order, 10, 10>
{
    [[nodiscard]] static constexpr Polynomial<double, order> operator()(
        [[maybe_unused]] const EFTFormFactorInputData& input_data,
        [[maybe_unused]] const NuclearResponseFormFactors<basis, order>& nuclear_form_factors) noexcept
    {
        const double coeff = 0.25*input_data.quad_inverse_b_sq;
        return Monomial<double, 1>{coeff}*nuclear_form_factors.sigma2;
    }
};

template <IsospinBasis basis, std::size_t order>
struct EFTFormFactorInitHelper<basis, order, 11, 11>
{
    [[nodiscard]] static constexpr Polynomial<double, order> operator()(
        const EFTFormFactorInputData& input_data,
        [[maybe_unused]] const NuclearResponseFormFactors<basis, order>& nuclear_form_factors) noexcept
    {
        const double coeff = 0.25*input_data.eft_spin_factor*input_data.quad_inverse_b_sq;
        return Monomial<double, 1>{coeff}*nuclear_form_factors.m;
    }
};

template <IsospinBasis basis, std::size_t order>
struct EFTFormFactorInitHelper<basis, order, 1, 3>
{
    [[nodiscard]] static constexpr Polynomial<double, order> operator()(
        const EFTFormFactorInputData& input_data,
        [[maybe_unused]] const NuclearResponseFormFactors<basis, order>& nuclear_form_factors) noexcept
    {
        const double coeff = 0.5*input_data.inverse_nucleus_mass*input_data.quad_inverse_b_sq;
        return Monomial<double, 1>{coeff}*nuclear_form_factors.m_phi2;
    }
};

template <IsospinBasis basis, std::size_t order>
struct EFTFormFactorInitHelper<basis, order, 4, 5>
{
    [[nodiscard]] static constexpr Polynomial<double, order> operator()(
        const EFTFormFactorInputData& input_data,
        [[maybe_unused]] const NuclearResponseFormFactors<basis, order>& nuclear_form_factors) noexcept
    {
        const double coeff
            = -0.125*input_data.eft_spin_factor
                *input_data.inverse_nucleus_mass*input_data.quad_inverse_b_sq;
        return Monomial<double, 1>{coeff}*nuclear_form_factors.delta_sigma1;
    }
};

template <IsospinBasis basis, std::size_t order>
struct EFTFormFactorInitHelper<basis, order, 4, 6>
{
    [[nodiscard]] static constexpr Polynomial<double, order> operator()(
        const EFTFormFactorInputData& input_data,
        [[maybe_unused]] const NuclearResponseFormFactors<basis, order>& nuclear_form_factors) noexcept
    {
        const double coeff = 0.0625*input_data.eft_spin_factor*input_data.quad_inverse_b_sq;
        return Monomial<double, 1>{coeff}*nuclear_form_factors.sigma2;
    }
};

template <IsospinBasis basis, std::size_t order>
struct EFTFormFactorInitHelper<basis, order, 8, 9>
{
    [[nodiscard]] static constexpr Polynomial<double, order> operator()(
        const EFTFormFactorInputData& input_data,
        [[maybe_unused]] const NuclearResponseFormFactors<basis, order>& nuclear_form_factors) noexcept
    {
        const double coeff
            = 0.125*input_data.eft_spin_factor
                *input_data.inverse_nucleus_mass*input_data.quad_inverse_b_sq;
        return Monomial<double, 1>{coeff}*nuclear_form_factors.delta_sigma1;
    }
};

template <IsospinBasis basis, std::size_t order, std::size_t I, std::size_t J>
static constexpr auto eft_form_factor_init = EFTFormFactorInitHelper<basis, order, I, J>{};

} // namespace detail

/**
    @brief Class for nonrelativistic nuclear effective theory form factors.

    @tparam basis_param Choice of isospin basis for the form factors.
    @tparam order Polynomial order in squared momentum transfer.
    @tparam I Effective theory operator index.
    @tparam J Effective theory operator index.

    This class encodes the dependence of the form factors
    \f$F_{I,J}^{(N,N')}(y)\f$ of the nonerlativistic effective theory of DM
    nuclear interactions on the nuclear response form factors. Although these
    form factors are typically expressed as functions of the squared momentum
    transfer \f$q^2\f$, this class expresses them in terms of the parameter
    \f&y = (bq/2)^2\f$, where \f$b\f$ is the harmonic oscillator parameter.
    This is consistent with the fact that the effective theory form factors are
    expressed in terms of the nuclear form factors \f$F_K^{(N,N')}(y)\f$, where
    \f$K\f$ is one of \f$\{M, \Delta, \Sigma', \Sigma'', Delta\Sigma', \Phi'',
    M\Phi''\}\f$.

    Each form factor can in general be written as a linear function of
    \f$v_\perp^2\f$, that is
    \f[
        F_{I,J}^{(N,N')} = F_{I,J,0}^{(N,N')} + v_\perp^2F_{I,J,1}^{(N,N')}.
    \f]
    Furthermore, since \f$F_K^{(N,N')}(y)\f$ are generally of the form
    \f[
        F_K^{(N,N')}(y) = e^{-y}P_K^{(N,N)}(y),
    \f]
    where \f$P_K^{(N,N)}(y)\f$ is a polynomial, what this class stores are
    in fact the polynomials
    \f[
        e^yF_{I,J,0}^{(N,N')},\qquad e^yF_{I,J,1}^{(N,N')}.
    \f]

    The form factors are given by [1]
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
private:
    [[nodiscard]] static consteval TransverseVelocityDependence
    evaluate_transverse_velocity_dependence() noexcept
    {
        if constexpr (I == J && J == 7)
            return TransverseVelocityDependence::mono;
        else if constexpr (I == J && (J == 3 || J == 5 || J == 8))
            return TransverseVelocityDependence::poly;
        else
            return TransverseVelocityDependence::none;
    }

public:
    static constexpr IsospinBasis basis = basis_param;
    static constexpr std::array<std::size_t, 2> indices = {I, J};
    static constexpr TransverseVelocityDependence transverse_velocity_dependence
        = evaluate_transverse_velocity_dependence();

    template <IsospinBasis basis>
    constexpr EFTFormFactor(
        [[maybe_unused]] ParticleSpin spin, [[maybe_unused]] double nucleus_mass,
        [[maybe_unused]] const NuclearResponseFormFactors<basis, order>& nuclear_form_factors) noexcept:
        m_polynomial{detail::eft_form_factor_init<basis, order, I, J>(spin, nucleus_mass, nuclear_form_factors)} {}

    /**
        @brief Tells whether the form factor depends on DM spin.
    */
    [[nodiscard]] static consteval bool is_spin_dependent() noexcept
    {
        return ((I == 4) || (I == 5) || (I == 6) || (I == 8) || (I == 9) || (I == 11))
            && ((J == 4) || (J == 5) || (J == 6) || (J == 8) || (J == 9) || (J == 11));
    }

    /**
        @brief Retrieve given component of the form factor.

        @tparam transverse_order Order of the component in squared transverse
        velocity.
    */
    template <std::size_t transverse_order>
        requires (transverse_order < 2)
    [[nodiscard]] constexpr const Polynomial<double, order>&
    component() const noexcept
    {
        if constexpr (transverse_order == 0)
            return nontransverse_component();
        else
            return transverse_component();
    }

private:
    static constexpr Polynomial<double, order> zero_polynomial = {};

    [[nodiscard]] constexpr const Polynomial<double, order>&
    nontransverse_component() const noexcept
    {
        if constexpr (transverse_velocity_dependence == TransverseVelocityDependence::mono)
            return zero_polynomial;
        else if constexpr (transverse_velocity_dependence == TransverseVelocityDependence::none)
            return m_polynomial;
        else
            return m_polynomial[0];
    }

    [[nodiscard]] constexpr const Polynomial<double, order>&
    transverse_component() const noexcept
    {
        if constexpr (transverse_velocity_dependence == TransverseVelocityDependence::mono)
            return m_polynomial;
        else if constexpr (transverse_velocity_dependence == TransverseVelocityDependence::none)
            return zero_polynomial;
        else
            return m_polynomial[1];
    }

    using StorageType = std::conditional<transverse_velocity_dependence == TransverseVelocityDependence::poly,
        std::array<Polynomial<double, order>, 2>,
        Polynomial<double, order>>;
    StorageType m_polynomial;
};

/**
    @brief The spin-averaged squared matrix element of the nonrelativistic
    effective theory.

    @tparam order order of the polynomial in squared momentum transfer.

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

    /**
        @brief Evaluate the spin-average squared matrix element at a given
        momentum transfer.
    */
    [[nodiscard]] std::array<double, 2>
    operator()(double momentum_transfer) const noexcept
    {
        const double q_sq = momentum_transfer*momentum_transfer;
        const double y = 0.25*q_sq*m_b_sq;

        const double decay = exponential(-2.0*y);
        return {decay*m_polynomial[0](y), decay*m_polynomial[1](y)};
    }

    /**
        @brief Retrieve given component of the spin-averaged squared matrix element.

        @tparam transverse_order Order of the component in squared transverse
        velocity.
    */
    template <std::size_t transverse_order>
        requires (transverse_order < 2)
    [[nodiscard]] constexpr const Polynomial<double, order>&
    component() const noexcept
    {
        return m_polynomial[transverse_order];
    }

private:
    /**
        @brief Generate coefficients of Taylor polynomial of exponential
        function.
    */
    template <std::size_t taylor_order>
    [[nodiscard]] static consteval Polynomial<double, taylor_order>
    exp_taylor_polynomial() noexcept
    {
        Polynomial<double, taylor_order> res{};
        res[0] = 1.0;
        for (std::size_t i = 1; i <= order; ++i)
            res[i] = res[i - 1]/double(i);
        return res;
    }

    template <typename T>
    [[nodiscard]] double exponential(double x) noexcept
    {
        if constexpr (std::same_as<T, int>)
            return std::exp(x);
        else
            return exp_taylor_polynomial<order>()(x);
    }

    std::array<Polynomial<double, order>, 2> m_polynomial;
    double m_b_sq{};
};

/**
    @brief Class for describing DM interactions in the nonrelativistc effective
    theory of DM nuclear interactions.

    @tparam basis_param Choice of isospin basis for the form factors.
*/
template <IsospinBasis basis_param, std::size_t order, typename... FormFactorTypes>
class DMInteraction
{
public:
    static constexpr IsospinBasis basis = basis_param;

    DMInteraction(
        ParticleSpin spin, Isotope isotope,
        const NuclearResponseFormFactors<basis, order>& nuclear_form_factors):
        m_form_factors{FormFactorTypes{detail::EFTFormFactorInputData{spin, isotope}, nuclear_form_factors}...},
        m_b_sq{harmonic_oscillator_parameter_sq(isotope.mass_number)} {}

    /**
        @brief Generate spin-averaged squared matrix-element.

        @param wilson_coefficients

        @return Function-like object representing the spin-averaged squared
        matrix element.
    */
    [[nodiscard]] constexpr InteractionFactor<order>
    operator()(zest::MDSpan<const double, 15, 2> wilson_coefficients) const noexcept
    {
        return std::apply([&](FormFactorTypes... form_factors)
        {
            std::array<Polynomial<double, order>, 2> res{};
            ([&]{
                const auto& [I, J] = FormFactorTypes::indices;
                if constexpr (FormFactorTypes::transverse_velocity_dependence == TransverseVelocityDependence::poly)
                {
                    res[0] += form_factors.template component<0>()[0, 0]*wilson_coefficients[I - 1, 0]*wilson_coefficients[J - 1, 0]
                        + form_factors.template component<0>()[0, 1]*wilson_coefficients[I - 1, 0]*wilson_coefficients[J - 1, 1]
                        + form_factors.template component<0>()[1, 0]*wilson_coefficients[I - 1, 1]*wilson_coefficients[J - 1, 0]
                        + form_factors.template component<0>()[1, 1]*wilson_coefficients[I - 1, 1]*wilson_coefficients[J - 1, 1];
                    res[1] += form_factors.template component<1>()[0, 0]*wilson_coefficients[I - 1, 0]*wilson_coefficients[J - 1, 0]
                        + form_factors.template component<1>()[0, 1]*wilson_coefficients[I - 1, 0]*wilson_coefficients[J - 1, 1]
                        + form_factors.template component<1>()[1, 0]*wilson_coefficients[I - 1, 1]*wilson_coefficients[J - 1, 0]
                        + form_factors.template component<1>()[1, 1]*wilson_coefficients[I - 1, 1]*wilson_coefficients[J - 1, 1];
                }
                else if constexpr (FormFactorTypes::transverse_velocity_dependence == TransverseVelocityDependence::mono)
                {
                    res[1] += form_factors.template component<1>()[0, 0]*wilson_coefficients[I - 1, 0]*wilson_coefficients[J - 1, 0]
                        + form_factors.template component<1>()[0, 1]*wilson_coefficients[I - 1, 0]*wilson_coefficients[J - 1, 1]
                        + form_factors.template component<1>()[1, 0]*wilson_coefficients[I - 1, 1]*wilson_coefficients[J - 1, 0]
                        + form_factors.template component<1>()[1, 1]*wilson_coefficients[I - 1, 1]*wilson_coefficients[J - 1, 1];
                }
                else
                {
                    res[0] += form_factors.template component<0>()[0, 0]*wilson_coefficients[I - 1, 0]*wilson_coefficients[J - 1, 0]
                        + form_factors.template component<0>()[0, 1]*wilson_coefficients[I - 1, 0]*wilson_coefficients[J - 1, 1]
                        + form_factors.template component<0>()[1, 0]*wilson_coefficients[I - 1, 1]*wilson_coefficients[J - 1, 0]
                        + form_factors.template component<0>()[1, 1]*wilson_coefficients[I - 1, 1]*wilson_coefficients[J - 1, 1];
                }
            }(), ...);
            return InteractionFactor{m_b_sq, res};
        }, m_form_factors);
    }

private:
    std::tuple<FormFactorTypes...> m_form_factors;
    double m_b_sq;
};

} // namespace zdm::neft
