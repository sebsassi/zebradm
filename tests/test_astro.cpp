#include "astro.hpp"

#include <cmath>
#include <print>

bool is_close(double a, double b, double error)
{
    return std::abs(a - b) < error;
}

template <std::size_t N>
bool is_close(const std::array<double, N>& a, const std::array<double, N>& b, double error)
{
    bool res = true;
    for (std::size_t i = 0; i < N; ++i)
        res = res && is_close(a[i], b[i], error);
    return res;
}

template <std::size_t N>
bool is_close(const zdm::la::Vector<double, N>& a, const zdm::la::Vector<double, N>& b, double error)
{
    return is_close(std::array<double, N>(a), std::array<double, N>(b), error);
}

bool test_gcs_to_icrs_maps_z_nearly_to_north_galactic_pole(double error)
{
    const zdm::la::Vector ngp_pos = zdm::astro::orientation_km_2017.gcs_to_reference_cs()*zdm::la::Vector{0.0, 0.0, 1.0};
    const zdm::la::Vector true_ngp_pos = {
        std::cos(zdm::astro::orientation_km_2017.ngp_ra)*std::cos(zdm::astro::orientation_km_2017.ngp_dec),
        std::sin(zdm::astro::orientation_km_2017.ngp_ra)*std::cos(zdm::astro::orientation_km_2017.ngp_dec),
        std::sin(zdm::astro::orientation_km_2017.ngp_dec)
    };
    return is_close(ngp_pos, true_ngp_pos, error);
}

bool test_gcs_to_icrs_maps_x_approximately_to_sag_a_star_pos(double error)
{

    // Angular offset of Sagittarius A* from the galactic equator, using Sun's
    // galactic midplane offset and galactocentric distance froom Karim and
    // Mamajek (2017):
    // Karim, T. and Mamajek, E. E., “Revised geometric estimates of the North
    // Galactic Pole and the Sun's height above the Galactic mid-plane”,
    // Monthly Notices of the Royal Astronomical Society, vol. 465, no. 1,
    // OUP, pp. 472–481, 2017. doi:10.1093/mnras/stw2772.
    const double sag_a_star_offset = -std::atan2(17.0, 8200.0);

    const zdm::la::Vector approximate_sag_a_star_pos = zdm::astro::orientation_km_2017.gcs_to_reference_cs()*zdm::la::Vector{std::cos(sag_a_star_offset), 0.0, std::sin(sag_a_star_offset)};

    // Sagittarius A* position from Reid and Brunthaler (2004):
    // Reid, M. J. and Brunthaler, A., “The Proper Motion of Sagittarius A*.
    // II. The Mass of Sagittarius A*”, The Astrophysical Journal, vol. 616,
    // no. 2, IOP, pp. 872–884, 2004. doi:10.1086/424960.
    constexpr double sag_a_star_ra
        = 17.0*(2.0*std::numbers::pi/24.0)
        + 45.0*(2.0*std::numbers::pi/(24.0*60.0))
        + 40.0400*(2.0*std::numbers::pi/(24.0*3600.0));
    constexpr double sag_a_star_dec
        = -29.0*(2.0*std::numbers::pi/(360.0))
        - 28.138*(2.0*std::numbers::pi/(360.0*3600.0));
    const zdm::la::Vector true_sag_a_star_pos = {
        std::cos(sag_a_star_ra)*std::cos(sag_a_star_dec),
        std::sin(sag_a_star_ra)*std::cos(sag_a_star_dec),
        std::sin(sag_a_star_dec)
    };
    return is_close(approximate_sag_a_star_pos, true_sag_a_star_pos, error);
}

bool test_orbital_plane_to_reference_cs_gives_nearly_correct_orbital_plane_normal(
    const zdm::astro::OrbitOrientation& orientation, const zdm::la::Vector<double, 3> true_orbital_plane_normal, double error)
{
    const zdm::la::Vector orbital_plane_normal = orientation.orbital_plane_to_reference_cs()*zdm::la::Vector{0.0, 0.0, 1.0};
    return is_close(orbital_plane_normal, true_orbital_plane_normal, error);
}

bool test_orbital_plane_to_reference_cs_gives_nearly_correct_orbital_plane_x(
    const zdm::astro::OrbitOrientation& orientation, const zdm::la::Vector<double, 3> true_orbital_plane_x, double error)
{
    const zdm::la::Vector orbital_plane_x = orientation.orbital_plane_to_reference_cs()*zdm::la::Vector{1.0, 0.0, 0.0};
    std::println("{}\n{}", orbital_plane_x, true_orbital_plane_x);
    return is_close(orbital_plane_x, true_orbital_plane_x, error);
}

int main()
{
    assert(test_gcs_to_icrs_maps_z_nearly_to_north_galactic_pole(1.0e-15));
    assert(test_gcs_to_icrs_maps_x_approximately_to_sag_a_star_pos(1.0e-4));

    assert(test_orbital_plane_to_reference_cs_gives_nearly_correct_orbital_plane_normal(
        zdm::astro::OrbitOrientation{0.4, 0.0, 0.0}, {0.0, std::sin(-0.4), std::cos(-0.4)}, 1.0e-15));
    assert(test_orbital_plane_to_reference_cs_gives_nearly_correct_orbital_plane_normal(
        zdm::astro::OrbitOrientation{0.0, 1.5, 0.0}, {0.0, 0.0, 1.0}, 1.0e-15));
    assert(test_orbital_plane_to_reference_cs_gives_nearly_correct_orbital_plane_normal(
        zdm::astro::OrbitOrientation{0.0, 0.0, 2.3}, {0.0, 0.0, 1.0}, 1.0e-15));
    assert(test_orbital_plane_to_reference_cs_gives_nearly_correct_orbital_plane_normal(
        zdm::astro::OrbitOrientation{0.4, 1.5, 0.0}, {std::sin(-0.4)*std::cos(1.5 + 0.5*std::numbers::pi), std::sin(-0.4)*std::sin(1.5 + 0.5*std::numbers::pi), std::cos(0.4)}, 1.0e-15));
    assert(test_orbital_plane_to_reference_cs_gives_nearly_correct_orbital_plane_normal(
        zdm::astro::OrbitOrientation{0.4, 0.0, 2.3}, {0.0, std::sin(-0.4), std::cos(-0.4)}, 1.0e-15));

    assert(test_orbital_plane_to_reference_cs_gives_nearly_correct_orbital_plane_x(
        zdm::astro::OrbitOrientation{0.4, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0e-15));
    assert(test_orbital_plane_to_reference_cs_gives_nearly_correct_orbital_plane_x(
        zdm::astro::OrbitOrientation{0.0, 1.5, 0.0}, {1.0, 0.0, 0.0}, 1.0e-15));
    assert(test_orbital_plane_to_reference_cs_gives_nearly_correct_orbital_plane_x(
        zdm::astro::OrbitOrientation{0.0, 0.0, 2.3}, {std::cos(2.3), std::sin(2.3), 0.0}, 1.0e-15));
}
