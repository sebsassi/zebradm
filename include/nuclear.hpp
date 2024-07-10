#pragma once

#include <array>
#include <vector>
#include <span>
#include <cmath>

constexpr double harmonic_osc_param_sq(double mass_number)
{
    const double inv_cbrt = 1.0/std::cbrt(mass_number);
    return 1065.0/((45.0 - 25.0*inv_cbrt)*inv_cbrt);
}

constexpr std::array<double, 5> nuclear_response_coeff(double mass_number)
{
    const double mass = 0.93149410242*mass_number;
    const double inv_mass = 1.0/mass;
    return {1,0.5*inv_mass,0.5,0.5,inv_mass};
}

std::vector<double> form_factors(
    std::size_t momentum_degree, double mass_number,
    const std::span<double>& nuc_resp_coeff,
    const std::array<double, NI*NI*KMAX*KMAX*N_FF_CFF>& ff)
{
    constexpr std::size_t n_max = N_FF_CFF - 1;
    const std::size_t max_pow = std::min(momentum_degree, n_max);
    std::vector<double> form_factor(NI*NI*KMAX*KMAX*(max_pow + 1));
    const double damping = 0.25*harmonic_osc_param_sq(mass_number);

    double prefactor = 1.0;
    for (std::size_t n = 0; n <= max_pow; ++n)
    {
        double ecff = 1.0;
        for (std::size_t l = 0; l <= n; ++l)
        {
            for (std::size_t i = 0; i < NI*NI*KMAX*KMAX; ++i)
                form_factor[i*(max_pow + 1) + n] = ff[i*(max_pow + 1) + n - l]*ecff;
            ecff *= 2.0/double(l + 1);
        }
        for (std::size_t i = 0; i < NI*NI*KMAX*KMAX; ++i)
            form_factor[i*(max_pow + 1) + n] *= prefactor;
        prefactor *= damping;
    }

    return form_factor;
}