/*
Copyright (c) 2025, 2026 Sebastian Sassi

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

#include <numbers>
#include <random>

#include "celestial.hpp"
#include "coordinate_transforms.hpp"
#include "matrix.hpp"

namespace
{

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

template <std::size_t N, zdm::la::Action action, zdm::la::MatrixLayout layout>
bool is_close(const zdm::la::RotationMatrix<double, N, action, layout>& a, const zdm::la::RotationMatrix<double, N, action, layout>& b, double error)
{
    return is_close(std::array<double, N*N>(a), std::array<double, N*N>(b), error);
}

template <std::size_t N, zdm::la::Action action, zdm::la::MatrixLayout layout>
bool is_close(const zdm::la::RigidTransform<double, N, action, layout>& a, const zdm::la::RigidTransform<double, N, action, layout>& b, double error)
{
    return is_close(a.rotation(), b.rotation(), error) && is_close(a.translation(), b.translation(), error);
}

class TestTransform
{
public:
    using rigid_transform_type = zdm::la::RigidTransform<double, 3>;

    TestTransform(std::size_t seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> angle_dist{-std::numbers::pi, std::numbers::pi};
        std::uniform_real_distribution<double> vec_dist{-1.0, 1.0};
        m_transform
            = zdm::la::RigidTransform<double, 3>::from<chaining>(
                zdm::la::RotationMatrix<double, 3>::from_euler_angles<euler, chaining>(
                    angle_dist(rng), angle_dist(rng), angle_dist(rng)),
                zdm::la::Vector<double, 3>{vec_dist(rng), vec_dist(rng), vec_dist(rng)});
    }

    [[nodiscard]] constexpr zdm::la::RigidTransform<double, 3>
    operator()([[maybe_unused]] double t) const noexcept { return m_transform; }

    [[nodiscard]] constexpr zdm::la::RigidTransform<double, 3>
    operator()() const noexcept { return m_transform; }

private:
    static constexpr auto euler = zdm::la::EulerConvention::xyx;
    static constexpr auto chaining = zdm::la::Chaining::intrinsic;
    zdm::la::RigidTransform<double, 3> m_transform;
};

bool test_composite_with_inverse_gives_identity_transform([[maybe_unused]] double error)
{
    using InverseTestTransform = zdm::celestial::Inverse<TestTransform>;
    using Transform = zdm::celestial::Composite<
        zdm::la::Chaining::intrinsic, zdm::la::RigidTransform<double, 3>,
        TestTransform,
        InverseTestTransform
    >;

    return is_close(
        Transform{TestTransform{0UL}, InverseTestTransform{0UL}}(0.0),
        zdm::la::RigidTransform<double, 3>::identity(),
        error);
}

bool test_composite_order_is_correct(double error)
{
    constexpr auto chaining = zdm::la::Chaining::intrinsic;
    using Transform = zdm::celestial::Composite<
        chaining, zdm::la::RigidTransform<double, 3>,
        TestTransform,
        TestTransform,
        TestTransform
    >;

    return is_close(
        Transform{TestTransform{0UL}, TestTransform{1UL}, TestTransform{2UL}}(0.0),
        zdm::la::compose<chaining>(
            zdm::la::compose<chaining>(
                TestTransform{0}(0.0),
                TestTransform{1}(0.0)),
            TestTransform{2}(0.0)),
        error);
}

bool test_ecs_to_icrs_gives_correct_obliquity(double error)
{
    // J2000 obliquity from the IAU 2009 System of Astronomical Constants is
    // 84381.406 arc seconds.
    static constexpr double obliquity = 2.0*std::numbers::pi*84381.406/(60*60*360);
    const auto ydir = zdm::celestial::ECStoICRS{}()*zdm::la::Vector{0.0, 1.0, 0.0};
    return is_close(ydir[0], std::cos(obliquity), error)
        && is_close(ydir[1], std::sin(obliquity), error);
}

bool test_cirs_to_tirs_rotates_counterclockwise()
{
    const auto cirs_to_tirs = zdm::celestial::CIRStoTIRS{};
    const auto tirs_x0 = cirs_to_tirs(0.0)*zdm::la::Vector{1.0, 0.0, 0.0};
    const auto tirs_x = cirs_to_tirs(0.1)*zdm::la::Vector{1.0, 0.0, 0.0};
    return zdm::la::cross(zdm::la::swizzle<0, 1>(tirs_x0), zdm::la::swizzle<0, 1>(tirs_x)) > 0.0;
}

bool test_icrs_to_hcs_inverse_maps_z_to_lat_lon_direction(double error)
{
    const double lon = 1.2;
    const double lat = -0.5;
    const auto hcs_to_itrs = zdm::celestial::Inverse<zdm::celestial::ITRStoHCS>{lon, lat};
    const auto lon_lat_dir = zdm::coordinates::spherical_to_cartesian_geo(lon, lat);
    return is_close(hcs_to_itrs(0.0)(zdm::la::Vector{0.0, 0.0, 1.0}), lon_lat_dir, error);
}

} // namespace

int main()
{
    assert(test_composite_with_inverse_gives_identity_transform(1.0e-15));
    assert(test_composite_order_is_correct(1.0e-15));
    assert(test_ecs_to_icrs_gives_correct_obliquity(1.0e-15));
    assert(test_cirs_to_tirs_rotates_counterclockwise());
    assert(test_icrs_to_hcs_inverse_maps_z_to_lat_lon_direction(1.0e-15));
}
