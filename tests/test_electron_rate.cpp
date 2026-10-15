#include <numbers>

#include "electron_rate.hpp"
#include "radon_moments.hpp"
#include "types.hpp"
#include "units.hpp"

namespace
{

using namespace zdm::unit_symbols;

auto constant_dist_constant_resp_energy_differential_rate(
    zdm::QuantityOf<zdm::speed> auto max_dm_speed,
    zdm::QuantityOf<zdm::speed> auto lab_speed,
    zdm::QuantityOf<zdm::momentum_transfer> auto q_max,
    zdm::QuantityOf<zdm::mass> auto dm_mass,
    zdm::QuantityOf<zdm::energy> auto energy)
{
    const zdm::quantity top_speed = max_dm_speed + lab_speed;
    const zdm::quantity bot_speed = zdm::fabs(max_speed - lab_speed);
    const zdm::quantity min_speed_sq = 2*energy/dm_mass;

    const zdm::quantity q_top_min = dm_mass*(top_speed - zdm::sqrt(zdm::intpow<2>(top_speed) - min_speed_sq));
    const zdm::quantity q_top_max = dm_mass*(top_speed + zdm::sqrt(zdm::intpow<2>(top_speed) - min_speed_sq));
    const zdm::quantity q_bot_min = dm_mass*(bot_speed - zdm::sqrt(zdm::intpow<2>(bot_speed) - min_speed_sq));
    const zdm::quantity q_bot_max = dm_mass*(bot_speed + zdm::sqrt(zdm::intpow<2>(bot_speed) - min_speed_sq));

    const zdm::quantity q_top_min_norm = std::min{static_cast<double>(q_top_min/q_max), 1.0};
    const zdm::quantity q_top_max_norm = std::min{static_cast<double>(q_top_max/q_max), 1.0};
    const zdm::quantity q_bot_min_norm = std::min{static_cast<double>(q_bot_min/q_max), 1.0};
    const zdm::quantity q_bot_max_norm = std::min{static_cast<double>(q_bot_max/q_max), 1.0};

    std::pair antiderivatives = {
        [](double q_norm)
        {
            return 0.0;
        },
        [](double q_norm)
        {
            return 0.0;
        },
    };

    const double integrated_weighted_radon_transform
        = (2*zdm::intpow<2>(std::numbers::pi)*zdm::intpow<3>(q_max)/lab_speed)
        *(antiderivatives.first(q_top_max) - antiderivatives.first(q_bot_max)
        + antiderviatives.second(q_bot_max) - antiderivatives.second(q_bot_min)
        + antiderviatives.first(q_bot_min) - antiderivatives.first(q_top_min));

}


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
