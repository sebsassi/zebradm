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
#include "radon_moments.hpp"
#include "moments.hpp"

#include <cassert>

namespace zdm::detail
{

void transverse_radon_moments(
    IsotropicZernikeSpan<const double> in_radon,
    IsotropicZernikeSpan<const double> in_r2_radon,
    IsotropicRadonMomentSpan<double, MomentSet::transverse> out)
{
    if (in_radon.order() < 3) return;
    assert(in_r2_radon.order() == in_radon.order() + 2);
    assert(in_r2_radon.order() + 2 <= out.max_order());

    out[IsoMoment::quadratic, 0] = in_r2_radon[0] - (2.0/3.0)*in_radon[0] - (4.0/15.0)*in_radon[2];
    out[IsoMoment::linear, 0] = in_radon[0] + (2.0/5.0)*in_radon[2];
    out[IsoMoment::identity, 0] = in_radon[0];

    const std::size_t nmax = (in_r2_radon.order() - 1) & (~1UL);
    for (std::size_t n = 2; n < nmax - 2; n += 2)
    {
        const auto dn = double(n);
        out[IsoMoment::quadratic, n] = in_r2_radon[n] - 2.0*(in_radon[n - 2]*dn*(dn - 1.0)/((2.0*dn - 3.0)*(2.0*dn - 1.0))
            + in_radon[n]*((dn + 1.0)*(dn + 1.0)/(2.0*dn + 3.0) + dn*dn/(2.0*dn - 1.0))/(2.0*dn + 1.0)
            + in_radon[n + 2]*(dn + 1.0)*(dn + 2.0)/((2.0*dn + 3.0)*(2.0*dn + 5.0)));
        out[IsoMoment::linear, n] = in_radon[n]*(dn + 1.0)/(2.0*dn + 1.0) + in_radon[n + 2]*(dn + 2.0)/(2.0*dn + 5.0);
        out[IsoMoment::identity, n] = in_radon[n];
    }

    const auto dn = double(nmax);
    out[IsoMoment::quadratic, nmax - 2] = in_r2_radon[nmax - 2] - 2.0*(in_radon[nmax - 4]*(dn - 2.0)*(dn - 3.0)/((2.0*dn - 7.0)*(2.0*dn - 5.0))
        + in_radon[nmax - 2]*((dn - 1.0)*(dn - 1.0)/(2.0*dn - 1.0) + (dn - 2.0)*(dn - 2.0)/(2.0*dn - 5.0))/(2.0*dn - 3.0));
    out[IsoMoment::linear, nmax - 2] = in_radon[nmax - 2]*(dn - 1.0)/(2.0*dn - 3.0);
    out[IsoMoment::identity, nmax - 2] = in_radon[nmax - 2];

    out[IsoMoment::quadratic, nmax] = in_r2_radon[nmax] - in_radon[nmax - 2]*2.0*dn*(dn - 1.0)/((2.0*dn - 3.0)*(2.0*dn - 1.0));
    out[IsoMoment::linear, nmax] = 0.0;
    out[IsoMoment::identity, nmax] = 0.0;
}

}
