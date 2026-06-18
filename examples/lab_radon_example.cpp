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
#include "zebra_angle_integrator.hpp"

namespace
{

constexpr double reduced_mass(double m1, double m2)
{
    return m1*m2/(m1 + m2);
}

std::vector<double> calculate_vmin(
    double nucleus_mass, double dm_mass, double vesc, [[maybe_unused]] double vdisp, double emax)
{
    const double emin = 0.0;
    const double red_mass = reduced_mass(dm_mass, nucleus_mass);

    std::size_t count = 50;
    std::vector<double> vmin(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        const double energy = emin + (emax - emin)*(double(i)/double(count - 1UL));
        vmin[i] = std::sqrt(nucleus_mass*energy/(2.0*red_mass*red_mass));
        vmin[i] /= vesc;
    }

    return vmin;
}

// Left distribution unnormalized in this toy example
constexpr double distribution_norm([[maybe_unused]] double vdisp, [[maybe_unused]] double vesc)
{
    return 1.0;
}

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

zest::DynamicMDArray<double, 2> radon_transform(
    std::size_t dist_order, double vesc, double vdisp, double dm_mass,
    double nucleus_mass, std::span<double> times, double emax)
{
    const double dist_norm = distribution_norm(vdisp, vesc);
    auto velocity_distribution = [&](const std::array<double, 3>& v)
    {
        const double inv_vdisp = 1.0/vdisp;
        constexpr zdm::la::Matrix<double, 3, 3> sigma = {
            3.0, 1.4, 0.5,
            1.4, 0.3, 2.1,
            0.5, 2.1, 1.7
        };
        return dist_norm*std::exp(-0.5*zdm::la::quadratic_form(sigma, v)*(inv_vdisp*inv_vdisp));
    };

    const double lon = 0.0;
    const double lat = 1.5;
    const double vcirc = vdisp;

    zdm::celestial::GCStoHCS gcs_to_hcs{lon, lat, vcirc};

    std::vector<zdm::la::Vector<double, 3>> vlab{times.size()};
    for (std::size_t i = 0; i < times.size(); ++i)
        vlab[i] = zdm::celestial::transform_velocity(gcs_to_hcs, times[i]);

    const std::vector<double> vmin
        = calculate_vmin(nucleus_mass, dm_mass, vesc, vdisp, emax);

    // zebradm
    // zest
    const zest::zt::ZernikeExpansion dist_expansion
        = zest::zt::ZernikeTransformerNormalGeo{}.forward_transform(
            velocity_distribution, vesc, dist_order);

    zest::DynamicMDArray<double, 2> out{vlab.size(), vmin.size()};

    zdm::zebra::AngleIntegrator<zdm::DistType::aniso, zdm::RespType::iso>
    radon_integrator(dist_order);

    radon_integrator.integrate((typename decltype(dist_expansion)::const_view)(dist_expansion), vlab, vmin, out);

    for (auto& element : out.flatten())
        element *= vesc*vesc;

    return out;
}

} // namespace

int main([[maybe_unused]] int argc, char** argv)
{
    const double vmax = atof(argv[1]);
    const double vdisp = atof(argv[2]);
    const std::size_t dist_order = std::size_t(atoi(argv[3]));
    const double dm_mass = atof(argv[4]);
    const double nucleus_mass = atof(argv[5]);
    const std::string_view start_date{argv[6]};
    const std::string_view end_date{argv[7]};
    const std::size_t ntime = std::size_t(atoi(argv[8]));
    const double emax = atof(argv[9]);

    std::vector<double> times = zdm::time::ut1_interval<zdm::time::j2000_utc>(start_date, end_date, ntime, "%Y-%m-%d").value();

    const auto out = radon_transform(dist_order, vmax, vdisp, dm_mass, nucleus_mass, times, emax);

    //const char* fname = "lab_radon_example.dat";
    //print_to_file(fname, out);
    print_to_stdout((typename decltype(out)::const_view)(out));
}
