#include <numbers>

#include "electron_rate.hpp"
#include "mp-units/systems/hep/constants.h"
#include "radon_moments.hpp"
#include "types.hpp"
#include "units.hpp"

namespace
{

using namespace zdm::unit_symbols;

auto constant_dist_constant_resp_energy_differential_rate(
    zdm::QuantityOf<zdm::speed> auto max_dm_speed,
    zdm::QuantityOf<zdm::mass> auto dm_mass,
    zdm::QuantityOf<zdm::energy_density> auto dm_energy_density,
    zdm::QuantityOf<zdm::cross_section> auto dm_electron_cross_section,
    zdm::QuantityOf<zdm::momentum_transfer> auto q_max,
    zdm::QuantityOf<zdm::mass_density> auto target_mass_density,
    zdm::QuantityOf<zdm::speed> auto lab_speed,
    zdm::QuantityOf<zdm::energy> auto energy)
{
    const zdm::quantity top_speed = max_dm_speed + lab_speed;
    const zdm::quantity bot_speed = zdm::abs(max_dm_speed - lab_speed);
    const zdm::quantity min_speed_sq = 2*energy/dm_mass;

    const zdm::quantity q_top_min = dm_mass*(top_speed - zdm::sqrt(zdm::intpow<2>(top_speed) - min_speed_sq));
    const zdm::quantity q_top_max = dm_mass*(top_speed + zdm::sqrt(zdm::intpow<2>(top_speed) - min_speed_sq));
    const zdm::quantity q_bot_min = dm_mass*(bot_speed - zdm::sqrt(zdm::intpow<2>(bot_speed) - min_speed_sq));
    const zdm::quantity q_bot_max = dm_mass*(bot_speed + zdm::sqrt(zdm::intpow<2>(bot_speed) - min_speed_sq));

    const double q_top_min_norm = std::min(static_cast<double>((q_top_min/q_max).in(zdm::one)), 1.0);
    const double q_top_max_norm = std::min(static_cast<double>((q_top_max/q_max).in(zdm::one)), 1.0);
    const double q_bot_min_norm = std::min(static_cast<double>((q_bot_min/q_max).in(zdm::one)), 1.0);
    const double q_bot_max_norm = std::min(static_cast<double>((q_bot_max/q_max).in(zdm::one)), 1.0);

    const double e_norm = static_cast<double>((energy/(q_max*max_dm_speed)).in(zdm::one));
    const double m_norm = static_cast<double>((dm_mass*max_dm_speed/q_max).in(zdm::one));

    const double vlab_norm = static_cast<double>((lab_speed/max_dm_speed).in(zdm::one));

    std::pair antiderivatives = {
        [&](double q_norm)
        {
            const std::array coeffs = {
                (1.0/3.0)*zdm::intpow<3>(e_norm),
                -0.5*zdm::intpow<2>(e_norm)*vlab_norm,
                (e_norm/(2.0*m_norm) + zdm::intpow<2>(vlab_norm) - 1.0)*(1.0/3.0)*e_norm,
                (2.0/3.0 + vlab_norm - zdm::intpow<3>(vlab_norm)*(1.0/4.0) - e_norm*vlab_norm/m_norm)*(1.0/3.0),
                (zdm::intpow<2>(vlab_norm) + e_norm/(2.0*m_norm) - 1.0)*(1.0/10.0*m_norm),
                -vlab_norm/(6*zdm::intpow<2>(2.0*m_norm)),
                1.0/(7*zdm::intpow<3>(2.0*m_norm))
            };
            return q_norm*(
                    coeffs[0] + q_norm*coeffs[1]
                    + (coeffs[2] + q_norm*coeffs[3])*zdm::intpow<2>(q_norm)
                    + (coeffs[4] + q_norm*coeffs[5]
                        + coeffs[6]*zdm::intpow<2>(q_norm))*zdm::intpow<4>(q_norm));
        },
        [&](double q_norm)
        {
            const std::array coeffs = {
                -0.5*zdm::intpow<2>(e_norm),
                (2.0/3.0)*e_norm*vlab_norm,
                (1.0 - (4.0/3.0)*zdm::intpow<2>(vlab_norm) - e_norm/m_norm)*(1.0/4.0),
                vlab_norm/(5.0*m_norm),
                -1.0/(6.0*zdm::intpow<2>(2.0*m_norm))
            };
            if (vlab_norm < 1.0)
                return 2.0*vlab_norm*zdm::intpow<2>(q_norm)*(
                    coeffs[0] + q_norm*coeffs[1]
                    + (coeffs[2] + q_norm*coeffs[3])*zdm::intpow<2>(q_norm)
                    + coeffs[4]*zdm::intpow<4>(q_norm));
            else
                return (1.0/3.0)*zdm::intpow<4>(q_norm);
        },
    };

    const double iwrt_prefactor = (2*zdm::intpow<2>(std::numbers::pi)/vlab_norm);
    const double integrated_weighted_radon_transform
        = iwrt_prefactor*(
            antiderivatives.first(q_top_max_norm) - antiderivatives.first(q_bot_max_norm)
            + antiderivatives.second(q_bot_max_norm) - antiderivatives.second(q_bot_min_norm)
            + antiderivatives.first(q_bot_min_norm) - antiderivatives.first(q_top_min_norm));

    const zdm::quantity event_rate_prefactor_numerator
        = dm_energy_density*dm_electron_cross_section*zdm::intpow<4>(q_max);
    const zdm::quantity event_rate_prefactor_denominator
        = (1.0*zdm::planck_constant*zdm::speed_of_light_in_vacuum)*(2.0*zdm::fine_structure_constant)
            *target_mass_density*dm_mass
            *zdm::intpow<2>(zdm::reduced_mass_of(dm_mass, 1.0*zdm::electron_mass))*max_dm_speed;
    const zdm::quantity event_rate_prefactor
        = event_rate_prefactor_numerator/event_rate_prefactor_denominator;

    static_assert(zdm::QuantityOf<decltype(event_rate_prefactor), zdm::energy_differential_rate_per_unit_mass>);
    return event_rate_prefactor*integrated_weighted_radon_transform;
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

    zest::DynamicMDArray<zdm::quantity<zdm::energy_differential_rate_per_unit_mass[zdm::one/(MeV*s*kg)]>, 2>
    reference{lab_velocities.size(), energies.size()};

    for (std::size_t i = 0; i < lab_velocities.size(); ++i)
        for (std::size_t j = 0; j < energies.size(); ++j)
            reference[i, j] = constant_dist_constant_resp_energy_differential_rate(
                max_speed,
                dm_mass,
                dm_energy_density,
                dm_electron_cross_section,
                max_momentum_transfer,
                target_density,
                zdm::la::norm(lab_velocities[i]),
                energies[j]).in(zdm::one/(MeV*s*kg));

    constexpr std::size_t order = 1;

    zdm::IsotropicRadonMomentArray<double, zdm::MomentSet::identity>
    velocity_dist_moments{order + 2};

    velocity_dist_moments[zdm::IsoMoment::identity, 0] = 1.0/3.0;
    velocity_dist_moments[zdm::IsoMoment::identity, 2] = -1.0/3.0;

    zdm::IsotropicZernikeExpansionVector<double> target_response{energies.size(), 1};
    for (std::size_t i = 0; i < energies.size(); ++i)
        target_response[i, 0] = 1.0/std::numbers::sqrt3;

    zest::DynamicMDArray<zdm::quantity<zdm::energy_differential_rate_per_unit_mass[zdm::one/(MeV*s*kg)]>, 2>
    out{lab_velocities.size(), energies.size()};

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
