/*
Copyright (c) 2025 Sebastian Sassi

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

[[nodiscard]] inline double
harmonic_oscillator_parameter_sq(std::size_t atomic_number) noexcept
{
    constexpr double a = 0.0;
    constexpr double b1 = 0.0;
    constexpr double b2 = 0.0;
    const double cbrt_a = std::cbrt(double(atomic_number));
    return cbrt_a*cbrt_a*a/(b1*cbrt_a - b2);
}

template <std::size_t Order>
struct NuclearResponseFormFactors
{
    zest::MDArray<Polynomial<double, Order>, 2, 2> m;
    zest::MDArray<Polynomial<double, Order>, 2, 2> delta;
    zest::MDArray<Polynomial<double, Order>, 2, 2> sigma1;
    zest::MDArray<Polynomial<double, Order>, 2, 2> sigma2;
    zest::MDArray<Polynomial<double, Order>, 2, 2> delta_sigma1;
    zest::MDArray<Polynomial<double, Order>, 2, 2> phi1;
    zest::MDArray<Polynomial<double, Order>, 2, 2> phi2;
    zest::MDArray<Polynomial<double, Order>, 2, 2> phi2_m;
};

template <std::size_t Order, std::size_t I, std::size_t J>
class EFTFormFactor {};

template <std::size_t Order>
class EFTFormFactor<Order, 0, 0>
{
public:
    static constexpr std::array<std::size_t, 2> indices = {0, 0};

    constexpr EFTFormFactor(
        double nucleus_mass,
        const NuclearResponseFormFactors<Order>& nuclear_form_factors) noexcept:
        m_polynomial{init(nucleus_mass, nuclear_form_factors)} {}

private:
    [[nodiscard]] static constexpr std::array<Polynomial<double, Order>, 2>
    init(
        [[maybe_unused]] double nucleus_mass,
        const NuclearResponseFormFactors<Order>& nuclear_form_factors) noexcept
    {
        return {
            nuclear_form_factors.m,
            Polynomial<double, Order>{}
        };
    }

    std::array<Polynomial<double, Order>, 2> m_polynomial;
};

template <std::size_t Order>
class InteractionFormFactor
{
public:
    InteractionFormFactor(const double b_sq, const std::array<Polynomial<double, Order>, 2>& polynomial):
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

    std::array<Polynomial<double, Order>, 2> m_polynomial;
    double m_b_sq{};
};

template <std::size_t Order, typename... FormFactorTypes>
class DMInteraction
{
public:
    DMInteraction(Isotope isotope, const NuclearResponseFormFactors<Order>& nuclear_form_factors):
        m_form_factors{FormFactorTypes{isotope.mass(), nuclear_form_factors}...},
        m_b_sq{harmonic_oscillator_parameter_sq(isotope.mass_number)} {}

    [[nodiscard]] constexpr InteractionFormFactor<Order>
    operator()(zest::MDSpan<const double, 15, 2> wilson_coefficients) const noexcept
    {
        return std::apply([&](FormFactorTypes... form_factors)
        {
            std::array<Polynomial<double, Order>, 2> res{};
            ([&]{
                const auto& [I, J] = FormFactorTypes::indices;
                res[0] += form_factors[0][0, 0]*wilson_coefficients[I, 0]*wilson_coefficients[J, 0]
                    + form_factors[0][0, 1]*wilson_coefficients[I, 0]*wilson_coefficients[J, 1]
                    + form_factors[0][1, 0]*wilson_coefficients[I, 1]*wilson_coefficients[J, 0]
                    + form_factors[0][1, 1]*wilson_coefficients[I, 1]*wilson_coefficients[J, 1];
                res[1] += form_factors[1][0, 0]*wilson_coefficients[I, 0]*wilson_coefficients[J, 0]
                    + form_factors[1][0, 1]*wilson_coefficients[I, 0]*wilson_coefficients[J, 1]
                    + form_factors[1][1, 0]*wilson_coefficients[I, 1]*wilson_coefficients[J, 0]
                    + form_factors[1][1, 1]*wilson_coefficients[I, 1]*wilson_coefficients[J, 1];
            }(), ...);
            return InteractionFormFactor{m_b_sq, res};
        }, m_form_factors);
    }

private:
    std::tuple<FormFactorTypes...> m_form_factors;
    double m_b_sq;
};

} // namespace zdm::neft
