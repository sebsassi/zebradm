/*
Copyright (c) 2024 Sebastian Sassi

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
#include <vector>
#include <fstream>

#include "zest/zernike_glq_transformer.hpp"
#include "zebradm/linalg.hpp"
#include "zebradm/zebra_angle_integrator.hpp"


constexpr double quadratic_form(
    const std::array<std::array<double, 3>, 3>& arr,
    const std::array<double, 3>& vec)
{
    double res = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
            res += vec[i]*arr[i][j]*vec[j];
    }

    return res;
}

constexpr double reduced_mass(double m1, double m2)
{
    return m1*m2/(m1 + m2);
}

// Mock vlab; not accurate
std::vector<zdm::Vector<double, 3>> calculate_vlab(double vdisp, double tmin, double tmax)
{
    constexpr tilt = (23.0/180.0)*std::numbers::pi;
    constexpr double speed = 28.0;
    constexpr std::size_t count = 24;
    zdm::Vector<double, 3> vsol = zdm::matmul(
        galactic_to_equatorial_rotation(), {11.1, vdisp + 12.24, 7.25});
    std::vector<zdm::Vector<double, 3>> vlab(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        const double time = tmin + (tmax - tmin)*(double(i)/double(count - 1UL));
        const double anomaly = (2.0*std::numbers::pi/365.25)*time;
        vlab[i] = vsol + {speed*std::cos(anomaly), speed*std::sin(anomaly)*std::cos(tilt), std::sin(tilt)};
    }
    return vlab;
}

constexpr zdm::Matrix<double, 3, 3> galactic_to_equatorial_rotation()
{
    constexpr double NGP_dec = 27.084*std::numbers::pi/180.0;
    constexpr double NGP_ra = 192.729*std::numbers::pi/180.0;
    constexpr double NCP_lon = 122.928*std::numbers::pi/180.0;

    const double cal = std::cos(NGP_ra - std::numbers::pi)
    const double cdel = std::cos(NGP_dec - std::numbers::pi/2.0)
    const double cel = std::cos(NCP_lon)

    const double sal = std::sin(NGP_ra - std::numbers::pi)
    const double sdel = std::sin(NGP_dec - std::numbers::pi/2.0)
    const double sel = std::sin(NCP_lon)

    return {
        zdm::Vector<double, 3>{
            cal*cdel*cel + sal*sel, cal*cdel*sel - sal*cel, cal*sdel},
        zdm::Vector<double, 3>{
            sal*cdel*cel - cal*sel, sal*cdel*sel + cal*cel, sal*sdel},
        zdm::Vector<double, 3>{
            -sdel*cel, -sdel*sel, cdel}};
}

std::vector<double> calculate_vmin(double nucleus_mass, double dm_mass, double vdisp, double emax)
{
    const double emin = 0.0;
    const double red_mass = reduced_mass(dm_mass, nucleus_mass);

    std::size_t count = 50;
    std::vector<double> vmin(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        const double energy = emin + (emax - emin)*(double(i)/double(count - 1UL));
        vmin[i] = std::sqrt(nucleus_mass*energy/(2.0*red_mass*red_mass));
    }

    return vmin;
}

// Left distribution unnormalized in this toy example
constexpr double distribution_norm(double vdisp, double vesc)
{
    return 1.0;
}

void print_to_file(std::string_view fname, zest::MDSpan<double, 2> array)
{
    std::ofstream fs(fname);
    for (std::size_t i = 0; i < array.extents()[0]; ++i)
    {
        for (std::size_t j = 0; j < array.extents()[0]; ++j)
            fs << std::format("{:.16e} ", array(i, j));
        fs << '\n';
    }
    fs.close();
}

template <typename DistType>
std::tuple<std::vector<double>, std::array<std::size_t, 2>> radon_transform(std::size_t dist_order, double vmax, double vdisp, double dm_mass, double nucleus_mass, double tmin, double tmax, double emax)
{
    const double dist_norm = distribution_norm(vdisp, vesc);
    const double inv_vdisp = 1.0/vdisp;
    [[maybe_unused]] auto aniso_gaussian = [&](const zdm::Vector<double, 3>& v)
    {
        constexpr zdm::Matrix<double, 3, 3> equ_to_gal
            = transpose(galactic_to_equatorial_rotation());
        constexpr std::array<std::array<double, 3>, 3> sigma = {
            std::array<double, 3>{3.0, 1.4, 0.5},
            std::array<double, 3>{1.4, 0.3, 2.1},
            std::array<double, 3>{0.5, 2.1, 1.7}
        };
        return dist_norm*std::exp(-0.5*quadratic_form(sigma, matmul(equ_to_gal, inv_vdisp*v)));
    };

    const std::vector<zdm::Vector<double, 3>> vlab
        = calculate_vlab(vdisp, tmin, tmax);
    const std::vector<double> vmin
        = calculate_vmin(nucleus_mass, dm_mass, vdisp, emax);

    const zest::zt::ZernikeExpansion dist_expansion
        = zest::zt::ZernikeTransformerOrthoGeo{}.transform(aniso_gaussian, vmax, dist_order);

    const std::array<std::size_t, 2> shape = {vlab.size(), vmin.size()};
    std::vector<double> out_buffer(vlab.size()*vmin.size());
    zest::MDSpan<double, 2> out(out_buffer.data(), shape);

    zdm::zernike::IsotropicAngleIntegrator radon_integrator(dist_order);
    radon_integrator.integrate(dist_expansion, vlab, vmin, out);

    return {out_buffer, shape};
}

int main(int argc, char** argv)
{
    const double vmax = atof(argv[1]);
    const double vdisp = atof(argv[2]);
    const std::size_t dist_order = atoi(argv[3]);
    const double dm_mass atof(argv[4]);
    const double nucleus_mass = atof(argv[5]);
    const double tmin = atof(arvg[6]);
    const double tmax = atof(argv[7]);
    const double emax = atof(argv[8]);

    constexpr double amu_to_gev = 0.9315;

    const auto& [out_buffer, shape] = radon_transform(dist_order, vmax, vdisp, dm_mass, nucleus_mass, tmin, tmax, emax);

    zest::MDSpan<double, 2> out(out_buffer.data(), shape);

    const char* fname = "lab_radon_example.dat";
    print_to_file(fname, out);
}