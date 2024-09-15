#pragma once

#include "zest/md_span.hpp"

void isotope_energy_differential_rate(
    zest::MDSpan<const std::array<double, 2>, 2> radon_transform, 
    std::span<const std::array<double, 2>> form_factors, double isotope_mass, double dm_mass, double local_density);

void energy_differential_rate(
    zest::MDSpan<const double, 2> radon_transform,
    std::span<const double> form_factors, double isotope_mass,
    double dm_mass, double local_density);