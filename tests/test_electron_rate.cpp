#include <numbers>

#include "electron_rate.hpp"
#include "radon_moments.hpp"
#include "types.hpp"

namespace
{

using namespace zdm::unit_symbols;

bool iso_iso_energy_differential_rate_is_correct_for_const_dist()
{
    constexpr zdm::quantity max_speed = zdm::speed(538.0*km/s);
    constexpr zdm::quantity dm_mass = zdm::mass(100.0*MeV/c2);
    constexpr zdm::quantity dm_energy_density = zdm::energy_density(0.3*GeV/cm3);
    constexpr zdm::quantity dm_electron_cross_section = zdm::cross_section(1.0e-40*cm2);
    constexpr zdm::quantity max_momentum_transfer = zdm::momentum_transfer(4.0*MeV/c);
    constexpr zdm::quantity target_density = zdm::mass_density(1.0*g/cm3);

    std::vector<zdm::quantity<zdm::energy[eV], double>> energies = {
        0.0*eV, 1.0*eV, 2.0*eV, 4.0*eV, 8.0*eV
    };
    std::vector<zdm::quantity<zdm::velocity[km/s], zdm::la::Vector<double, 3>>> lab_velocities = {
        zdm::velocity(zdm::la::Vector<double, 3>{135.0, -24.0, 112.0}*km/s),
        zdm::velocity(zdm::la::Vector<double, 3>{218.0, 3.0, -1.0}*km/s),
    };

    constexpr std::size_t order = 1;

    zdm::IsotropicRadonMomentArray<double, zdm::MomentSet::identity>
    velocity_dist_moments{order + 2};

    velocity_dist_moments[zdm::IsoMoment::identity, 0] = 1.0/3.0;
    velocity_dist_moments[zdm::IsoMoment::identity, 2] = -1.0/3.0;

    zdm::IsotropicZernikeExpansionVector<double> target_response{energies.size(), 1};
    for (std::size_t i = 0; i < energies.size(); ++i)
        target_response[i, 0] = 1.0/std::numbers::sqrt3;

    zest::DynamicMDArray<double, 2> out{lab_velocities.size(), energies.size()};

    zdm::ElectronRateCalculator<zdm::DistType::iso, zdm::RespType::iso>{}
        .energy_differential_rate(
            velocity_dist_moments, max_speed,
            dm_mass,
            dm_energy_density,
            dm_electron_cross_section,
            target_response, max_momentum_transfer,
            target_density,
            lab_velocities,
            energies,
            out);
}

} // namespace

int main()
{

}
