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
#include <format>
#include <fstream>
#include <print>
#include <vector>

#include "zest/zernike_glq_transformer.hpp"

#include "celestial.hpp"
#include "linalg.hpp"
#include "matrix.hpp"
#include "time.hpp"
#include "units.hpp"
#include "electron_rate.hpp"

namespace
{

using namespace zdm::unit_symbols;

// Left distribution unnormalized in this toy example
constexpr double distribution_norm(
    [[maybe_unused]] zdm::QuantityOf<zdm::speed> auto vdisp,
    [[maybe_unused]] zdm::QuantityOf<zdm::speed> auto vesc)
{
    return 1.0;
}

class SHM
{
public:
    constexpr SHM(
        zdm::QuantityOf<zdm::speed> auto vdisp, zdm::QuantityOf<zdm::speed> auto vesc) noexcept:
        m_velocity_dispersion{vdisp.template in<double>(km/s)},
        m_escape_velocity{vesc.template in<double>(km/s)},
        m_norm{normalization(vdisp, vesc)} {}

    [[nodiscard]] constexpr double
    operator()(zdm::QuantityOf<zdm::speed> auto v) noexcept
    {
        return m_norm*std::exp(-static_cast<double>((v/m_velocity_dispersion).in(zdm::one)));

    }

    [[nodiscard]] constexpr zdm::quantity<zdm::speed[km/s]>
    escape_velocity() const noexcept
    {
        return m_escape_velocity;
    }

    [[nodiscard]] constexpr zdm::quantity<zdm::speed[km/s]>
    velocity_dispersion() const noexcept
    {
        return m_velocity_dispersion;
    }

private:
    [[nodiscard]] static constexpr double
    normalization(zdm::QuantityOf<zdm::speed> auto vdisp, zdm::QuantityOf<zdm::speed> auto vesc) noexcept
    {
        const double ratio = static_cast<double>((vesc.template in<double>()/vdisp.template in<double>()).in(zdm::one));
        constexpr double standard_gaussian_norm = zdm::intpow<3>(std::numbers::inv_sqrtpi/std::numbers::sqrt2);
        const double gaussian_norm = standard_gaussian_norm*ratio;
        const double trunc_term = std::erf(ratio) - (2.0*std::numbers::inv_sqrtpi)*ratio*std::exp(-zdm::intpow<2>(ratio));
        return gaussian_norm/trunc_term;
    }

    zdm::quantity<zdm::speed[km/s]> m_velocity_dispersion;
    zdm::quantity<zdm::speed[km/s]> m_escape_velocity;
    double m_norm;
};

[[maybe_unused]]
void print_to_file(const char* fname, zest::DynamicMDSpan<const double, 2> array)
{
    std::ofstream fs(fname);
    for (std::size_t i = 0; i < array.extents()[0]; ++i)
    {
        for (std::size_t j = 0; j < array.extents()[1]; ++j)
            fs << std::format("{:.16e} ", array[i, j]);
        fs << '\n';
    }
    fs.close();
}

void print_to_stdout(zest::DynamicMDSpan<const double, 2> array)
{
    for (std::size_t i = 0; i < array.extent(0); ++i)
    {
        for (std::size_t j = 0; j < array.extent(1); ++j)
            std::print("{:.16e} ", array[i, j]);
         std::println("");
    }
}

// Arbitrary mock target response for demonstration.
static constexpr qmax = zdm::momentum_transfer(30.0*keV/c);
double response(zdm::quantity<zdm::energy[eV]> energy, zdm::quantity<zdm::momentum_transfer[keV/c]> q)
{
    const double q_norm = static_cast<double>(q/qmax);
    const double e_norm = static_cast<double>(energy/zdm::energy(50.0*eV));
    return std::exp(-e_norm)*zdm::intpow<2>(1.0 - q_norm);
}

template <zdm::QuantityOf<zdm::duration> Duration>
zest::DynamicMDArray<double, 2> radon_transform(
    std::size_t dist_order, SHM velocity_distribution, zdm::quantity<zdm::mass[GeV/c2]> dm_mass,
    std::span<zdm::quantity<zdm::duration[d]>> times, zdm::quantity<zdm::energy[eV]> emax,
    zdm::quantity<zdm::momentum_transfer[keV/c]> qmax,
    zdm::quantity<zdm::mass_density[g/cm3]> target_density)
{
    const double lon = 0.0;
    const double lat = 1.5;

    zdm::celestial::GCStoHCS gcs_to_hcs{lon, lat, velocity_distribution.velocity_dispersion()};

    using Velocity = zdm::quantity<zdm::velocity[zdm::si::kilo<zdm::meter>/zdm::second], zdm::la::Vector<double, 3>>;
    using Speed = zdm::quantity<zdm::speed[zdm::si::kilo<zdm::meter>/zdm::second]>;

    std::vector<Velocity> vlab{times.size()};
    for (std::size_t i = 0; i < times.size(); ++i)
        vlab[i] = zdm::celestial::transform_velocity(gcs_to_hcs, times[i]);

    const zdm::IsotropicZernikeExpansion dist_expansion
        = zdm::IsotropicZernikeTransformer{}
            .forward_transform(velocity_distribution, velocity_distribution.escape_velocity(), dist_order);

    const zdm::RadonMomentArray velocity_dist_moments
        = zdm::RadonTransformer{}.evaluate_transformed_moments(dist_expansion);

    std::vector<zdm::quantity<zdm::energy[eV]>> energies = zdm::linspace(zdm::energy(0.0*eV), emax, 30);

    zest::DynamicMDArray<double, 2> out{vlab.size(), energies.size()};

    const zdm::quantity dm_energy_density = zdm::energy_density(0.3*GeV/cm3);
    const zdm::quantity dm_electron_cross_section = zdm::cross_section(1.0e-40*cm2);

    zdm::ElectronRateCalculator<zdm::DistType::iso, zdm::RespType::iso>{}
        .energy_differential_rate(
            velocity_dist_moments, velocity_distribution.escape_velocity(),
            dm_mass,
            dm_energy_density,
            dm_electron_cross_section,
            target_response, qmax,
            target_density,
            vlab,
            energies,
            out);

    return out;
}

} // namespace

int main([[maybe_unused]] int argc, char** argv)
{
    using namespace zdm::unit_symbols;

    const zdm::quantity vmax = zdm::speed(atof(argv[1])*km/s);
    const zdm::quantity vdisp = zdm::speed(atof(argv[2])*km/s);
    const std::size_t dist_order = std::size_t(atoi(argv[3]));
    const zdm::quantity dm_mass = zdm::mass(atof(argv[4])*GeV/c2);
    const std::string_view start_date{argv[5]};
    const std::string_view end_date{argv[6]};
    const std::size_t ntime = std::size_t(atoi(argv[7]));
    const zdm::quantity emax = zdm::energy(atof(argv[8])*eV);
    const zdm::quantity qmax = zdm::momentum_transfer(atof(argv[9])*keV/c);
    const zdm::quantity target_density = zdm::mass_density(atof(argv[10])*g/cm3);

    auto times = zdm::time::ut1_interval<zdm::time::j2000_utc>(start_date, end_date, ntime, "%Y-%m-%d").value();

    const auto out = radon_transform(
        dist_order, SHM{vmax, vdisp}, dm_mass, std::span(times), emax, qmax, target_density);

    //const char* fname = "lab_radon_example.dat";
    //print_to_file(fname, out);
    print_to_stdout((typename decltype(out)::const_view)(out));
}
