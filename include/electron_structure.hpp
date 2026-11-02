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

#include "matrix.hpp"
#include "units.hpp"
#include "utility.hpp"

namespace zdm
{

namespace detail
{

[[nodiscard]] constexpr double
isotropic_electron_gas_eelf(double q, double e) noexcept
{

}

} // namespace detail

/**
    Electron energy loss spectrum for isotropic electron gas.

    @param fermi_momentum Fermi momentum of the gas.
    @param q Momentum transfer.
    @param e Energy transfer.

    This function gives the electron energy loss spectrum (EELS) for isotropic
    electron gas. The EELS is given by
    \f[
        W(\vec{q}, \omega)
            = \rm{Im}(\varepsilon(\vec{q}, \omega)^{-1})
            = \frac{4\pi}{q^2}\rm{Im}(\chi(\vec{q}, \omega)),
    \f]
    where \f$\varepsilon(\vec{q}, \omega)\f$ is the dielectric function, and
    \f$\chi(\vec{q}, \omega)\f$ ia the electric susceptibility. For isotropic
    three-dimensional electron gas, the susceptibility is given by the Lindhard
    function, with imaginary part
    \f[
        \chi(\vec{q}, \omega)
            = \frac{k_F^3}{8\pi E_F}\frac{1}{p}\begin{cases}
                0,\qquad
                    1 \leq \nu \text{ or } \nu \leq p(p - 2)
                        \text{ or } p(p + 2) \leq \nu,\\
                1 - \frac{1}{4}\left(\frac{\nu}{p} - p\right),\qquad
                    |p(p - 2)| \leq \nu \leq p(p + 2),\\
                \nu,\qquad p < 2 \text{ and } \nu < |p(p - 2)|,
            \end{cases}
    \f]
    where \f$\nu = \omega/k_F\f$ and \f$p = q/E_F\f$ are the energy and
    momentum transfer normalized by the Fermi energy and Fermi momentum,
    respectively.
*/
[[nodiscard]] constexpr double
isotropic_electron_gas_eels(
    QuantityOf<momentum> auto fermi_momentum,
    QuantityOf<momentum_transfer> auto q, QuantityOf<energy> auto e) noexcept
{
    const quantity fermi_energy = (util::intpow<2>(reduced_plank_constant)/(2.0*electron_mass))*util::intpow<2>(fermi_momentum);
    const auto physics_prefactor = static_cast<double>((speed_of_light_in_vacuum*fermi_momentum/fermi_energy).in(zdm::one));

    const auto q_norm = static_cast<double>((q/fermi_momentum).in(zdm::one));
    const auto e_norm = static_cast<double>((e/fermi_energy).in(zdm::one));

    const double inv_q_norm = 1.0/q_norm;

    constexpr double geometric_prefactor = 1.0/(8.0*std::numbers::pi);
    const double prefactor = geometric_prefactor*physics_prefactor;

    if (1.0 < e_norm || e_norm < q_norm*(q_norm - 2.0) || q_norm*(q_norm + 2.0) < e_norm)
        return 0.0;
    else if (-q_norm*(q_norm - 2.0) < e_norm)
        return prefactor*(1.0 - 0.25*(e_norm*inv_q_norm - q_norm))*util::intpow<3>(inv_q_norm);
    else
        return prefactor*e_norm*util::intpow<3>(inv_q_norm);
}

template <Quantity Mass>
[[nodiscard]] constexpr double
anisotropic_electron_gas_eels(
    QuantityOf<momentum> auto fermi_momentum, std::array<Mass, 3> masses,
    QuantityOf<momentum_transfer> auto q, QuantityOf<energy> auto e) noexcept
{
    const quantity mass_dos = mpu::cbrt(masses[0]*masses[1]*masses[2]);
    const la::Vector<double, 3> scale_factors = {
        std::sqrt(static_cast<double>(mass_dos/masses[0])),
        std::sqrt(static_cast<double>(mass_dos/masses[1])),
        std::sqrt(static_cast<double>(mass_dos/masses[2])),
    };
    const quantity q_scaled = scale_by(q, scale_factors);
    const auto rescale = static_cast<double>(dot(q, q)/dot(q_scaled, q_scaled));

    return rescale*anisotropic_electron_gas_eels(q_scaled, e);
}

} // namespace zdm
