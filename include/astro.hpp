#pragma once

#include <numbers>

#include "matrix.hpp"

namespace zdm
{

struct GalacticOrientation
{
    double ngp_dec;
    double ngp_ra;
    double ncp_lon;

    [[nodiscard]] constexpr la::RotationMatrix<double, 3>
    gcs_to_icrs() const noexcept
    {
        constexpr auto convention = la::EulerConvention::zyz;
        constexpr auto chaining = la::Chaining::intrinsic;

        return la::RotationMatrix<double, 3>::from_euler_angles<convention, chaining>(
                ncp_lon, 0.5*std::numbers::pi - ngp_dec, std::numbers::pi - ngp_ra);
    }
};

// Value of Karim and Mamajek (2017)
constexpr GalacticOrientation orientation_km_2017 = {
        27.084*std::numbers::pi/180.0,
        192.729*std::numbers::pi/180.0,
        122.928*std::numbers::pi/180.0
};

// Value of Schönrich, Binney, and Dehnen (2010)
constexpr la::Vector<double, 3> peculiar_velocity_sbd_2010 = {11.1, 12.24, 7.25};

} // namespace zdm 
