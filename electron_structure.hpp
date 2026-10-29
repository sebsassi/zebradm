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

namespace zdm
{

namespace detail
{

[[nodiscard]] constexpr double
isotropic_electron_gas_eelf(double q, double e) noexcept
{

}

}

[[nodiscard]] constexpr double
isotropic_electron_gas_eelf(
    QuantityOf<momentum> auto fermi_momentum,
    QuantityOf<momentum_transfer> auto q, QuantityOf<energy> auto e) noexcept
{
    const quantity fermi_energy = (intpow<2>(reduced_plank_constant)/(2.0*electron_mass))*intpow<2>(fermi_momentum);
    const double q_norm = static_cast<double>((q/fermi_momentum).in(zdm::one));
    const double e_norm = static_cast<double>((e/fermi_energy).in(zdm::one));
}

[[nodiscard]] constexpr double
anisotropic_electron_gas_eelf(
    QuantityOf<momentum> auto fermi_momentum, QuantityOf<mass> auto mass_dos,
    la::Matrix<double, 3, 3> dispersion_matrix,
    QuantityOf<momentum_transfer> auto q, QuantityOf<energy> auto e) noexcept
{

}

}
