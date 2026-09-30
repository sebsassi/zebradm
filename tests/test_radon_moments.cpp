#include "moments.hpp"
#include "radon_moments.hpp"

#include <print>

namespace
{

constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) <= tol*0.5*std::fabs(a + b) + tol;
}

bool test_isotropic_zernike_transverse_radon_helper_is_correct_for_constant_distribution(std::size_t order)
{
    zdm::IsotropicZernikeExpansion<double> expansion{order};
    expansion[0] = 1.0/std::numbers::sqrt3;

    zdm::zebra::IsotropicRadonMomentArray<double, zdm::zebra::MomentCategory::transverse>
    reference_moments{order + 4};

    reference_moments[zdm::zebra::IsoMoment::quadratic, 0] = 1.0/15.0;
    reference_moments[zdm::zebra::IsoMoment::linear, 0] = 1.0/5.0;
    reference_moments[zdm::zebra::IsoMoment::identity, 0] = 1.0/3.0;
    reference_moments[zdm::zebra::IsoMoment::quadratic, 2] = -5.0/21.0;
    reference_moments[zdm::zebra::IsoMoment::linear, 2] = -1.0/5.0;
    reference_moments[zdm::zebra::IsoMoment::identity, 2] = -1.0/3.0;
    reference_moments[zdm::zebra::IsoMoment::quadratic, 4] = 6.0/35.0;

    zdm::zebra::IsotropicRadonMomentArray<double, zdm::zebra::MomentCategory::transverse>
    moments{order + 4};

    zdm::zebra::RadonTransformer<zdm::DistType::iso, zdm::zebra::MomentCategory::transverse>{order}
        .evaluate_transformed_moments(expansion, zdm::zebra::IsotropicRadonMomentSpan(moments));

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t n : moments[zdm::zebra::IsoMoment::identity].indices())
        success = success
            && is_close(
                moments[zdm::zebra::IsoMoment::identity, n],
                reference_moments[zdm::zebra::IsoMoment::identity],
                tol);
    for (std::size_t n : moments[zdm::zebra::IsoMoment::linear].indices())
        success = success
            && is_close(
                moments[zdm::zebra::IsoMoment::linear, n],
                reference_moments[zdm::zebra::IsoMoment::linear],
                tol);
    for (std::size_t n : moments[zdm::zebra::IsoMoment::quadratic].indices())
        success = success
            && is_close(
                moments[zdm::zebra::IsoMoment::quadratic, n],
                reference_moments[zdm::zebra::IsoMoment::quadratic],
                tol);

    if (!success)
    {
        std::println("components reference");
        for (std::size_t n : moments.indices())
            std::println("[{}, {}, {}] [{}, {}, {}]",
                    moments[zdm::zebra::IsoMoment::identity, n],
                    moments[zdm::zebra::IsoMoment::linear, n],
                    moments[zdm::zebra::IsoMoment::quadratic, n],
                    reference_moments[zdm::zebra::IsoMoment::identity, n],
                    reference_moments[zdm::zebra::IsoMoment::linear, n],
                    reference_moments[zdm::zebra::IsoMoment::quadratic, n]);
    }

    return success;
}

bool test_isotropic_zernike_transverse_radon_helper_components_are_consistent(std::size_t order)
{
    zdm::IsotropicZernikeExpansion<double> expansion{order};
    for (auto& element : expansion.flatten())
        element = 1.0;

    zdm::IsotropicZernikeExpansion<double> r2_expansion{order + 2};

    zdm::IsotropicZernikeExpansion<double> radon{order + 2};
    zdm::zebra::radon_transform(expansion, radon);

    zdm::IsotropicZernikeExpansion<double> r2_radon{order + 4};

    const zdm::zebra::detail::ZernikeRecursionData recursion_data{order + 4};
    zdm::zebra::detail::multiply_by_r2_and_radon_transform_inplace(
        recursion_data, zdm::IsotropicZernikeSpan<const double>(expansion), r2_radon);

    zdm::zebra::IsotropicRadonMomentArray<double, zdm::zebra::MomentCategory::transverse>
    composite_moments{order + 4};

    zdm::zebra::detail::transverse_radon_components(radon, r2_radon, composite_moments);

    zdm::IsotropicZernikeExpansion<double, 3> direct_moments{order + 4};
    zdm::zebra::RadonTransformer<zdm::DistType::iso, zdm::zebra::MomentCategory::transverse>{order}
        .evaluate_transformed_moments(expansion, direct_moments);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t n : moments[zdm::zebra::IsoMoment::identity].indices())
        success = success
            && is_close(
                direct_moments[zdm::zebra::IsoMoment::identity, n],
                composite_moments[zdm::zebra::IsoMoment::identity],
                tol);
    for (std::size_t n : moments[zdm::zebra::IsoMoment::linear].indices())
        success = success
            && is_close(
                direct_moments[zdm::zebra::IsoMoment::linear, n],
                composite_moments[zdm::zebra::IsoMoment::linear],
                tol);
    for (std::size_t n : moments[zdm::zebra::IsoMoment::quadratic].indices())
        success = success
            && is_close(
                direct_moments[zdm::zebra::IsoMoment::quadratic, n],
                composite_moments[zdm::zebra::IsoMoment::quadratic],
                tol);

    if (!success)
    {
        std::println("direct composite");
        for (std::size_t n : moments.indices())
            std::println("[{}, {}, {}] [{}, {}, {}]",
                    direct_moments[zdm::zebra::IsoMoment::identity, n],
                    direct_moments[zdm::zebra::IsoMoment::linear, n],
                    direct_moments[zdm::zebra::IsoMoment::quadratic, n],
                    composite_moments[zdm::zebra::IsoMoment::identity, n],
                    composite_moments[zdm::zebra::IsoMoment::linear, n],
                    composite_moments[zdm::zebra::IsoMoment::quadratic, n]);
    }

    return success;
}

bool test_isotropic_zernike_transverse_radon_helper_components_are_consistent(std::size_t order, std::size_t index)
{
    zdm::IsotropicZernikeExpansion<double> expansion{order};
    expansion[index] = 1.0;

    zdm::IsotropicZernikeExpansion<double> r2_expansion{order + 2};

    zdm::IsotropicZernikeExpansion<double> radon{order + 2};
    zdm::zebra::radon_transform(expansion, radon);

    zdm::IsotropicZernikeExpansion<double> r2_radon{order + 4};

    const zdm::zebra::detail::ZernikeRecursionData recursion_data{order + 4};
    zdm::zebra::detail::multiply_by_r2_and_radon_transform_inplace(
        recursion_data, zdm::IsotropicZernikeSpan<const double>(expansion), r2_radon);

    zdm::zebra::IsotropicRadonMomentArray<double, zdm::zebra::MomentCategory::transverse>
    composite_moments{order + 4};
    zdm::zebra::detail::transverse_radon_components(radon, r2_radon, composite_moments);

    zdm::zebra::IsotropicRadonMomentArray<double, zdm::zebra::MomentCategory::transverse>
    direct_moments{order + 4};
    zdm::zebra::RadonTransformer<zdm::DistType::iso, zdm::zebra::MomentCategory::transverse>{order}
        .evaluate_transverse_components(expansion, direct_components);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t n : moments[zdm::zebra::IsoMoment::identity].indices())
        success = success
            && is_close(
                direct_moments[zdm::zebra::IsoMoment::identity, n],
                composite_moments[zdm::zebra::IsoMoment::identity],
                tol);
    for (std::size_t n : moments[zdm::zebra::IsoMoment::linear].indices())
        success = success
            && is_close(
                direct_moments[zdm::zebra::IsoMoment::linear, n],
                composite_moments[zdm::zebra::IsoMoment::linear],
                tol);
    for (std::size_t n : moments[zdm::zebra::IsoMoment::quadratic].indices())
        success = success
            && is_close(
                direct_moments[zdm::zebra::IsoMoment::quadratic, n],
                composite_moments[zdm::zebra::IsoMoment::quadratic],
                tol);

    if (!success)
    {
        std::println("direct composite");
        for (std::size_t n : direct_moments.indices())
            std::println("[{}, {}, {}] [{}, {}, {}]",
                    direct_moments[zdm::zebra::IsoMoment::identity, n],
                    direct_moments[zdm::zebra::IsoMoment::linear, n],
                    direct_moments[zdm::zebra::IsoMoment::quadratic, n],
                    composite_moments[zdm::zebra::IsoMoment::identity, n],
                    composite_moments[zdm::zebra::IsoMoment::linear, n],
                    composite_moments[zdm::zebra::IsoMoment::quadratic, n]);
    }

    return success;
}

} // namespace

int main()
{
    assert(test_isotropic_zernike_transverse_radon_helper_is_correct_for_constant_distribution(10));

    assert(test_isotropic_zernike_transverse_radon_helper_components_are_consistent(20, 0));
    assert(test_isotropic_zernike_transverse_radon_helper_components_are_consistent(20, 2));
    assert(test_isotropic_zernike_transverse_radon_helper_components_are_consistent(20, 4));

    assert(test_isotropic_zernike_transverse_radon_helper_components_are_consistent(20));
    assert(test_isotropic_zernike_transverse_radon_helper_components_are_consistent(21));
}
