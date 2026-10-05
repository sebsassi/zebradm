#include "moments.hpp"
#include "radon_moments.hpp"

#include <print>

namespace
{

constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) <= tol*0.5*std::fabs(a + b) + tol;
}

bool test_radon_transformer_iso_trans_is_correct_for_constant_distribution(std::size_t order)
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
        .evaluate_transformed_moments(expansion, moments);

    constexpr double tol = 1.0e-13;

    const std::array moment_labels = {
        zdm::zebra::IsoMoment::identity,
        zdm::zebra::IsoMoment::linear,
        zdm::zebra::IsoMoment::quadratic,
    };

    bool success = true;
    for (auto label : moment_labels)
    {
        for (std::size_t n : moments[label].indices())
            success = success && is_close(moments[label, n], reference_moments[label, n], tol);
    }

    if (!success)
    {
        for (auto label : moment_labels)
        {
            std::println("{}: moment reference", zdm::zebra::to_string(label));
            for (std::size_t n : moments[label].indices())
                std::println("{}: {} {}", n, moments[label, n], reference_moments[label, n]);
        }
    }

    return success;
}

bool test_radon_transformer_aniso_trans_is_correct_for_constant_distribution(std::size_t order)
{
    zdm::ZernikeExpansion<double> expansion{order};
    expansion[0, 0, 0, 0] = 1.0/std::numbers::sqrt3;

    zdm::zebra::RadonMomentArray<double, zdm::zebra::MomentCategory::transverse>
    reference_moments{order + 4};

    constexpr double sqrt5 = 2.2360679774997896964091737;
    reference_moments[zdm::zebra::Moment::identity, 0, 0, 0, 0] = 1.0/3.0;
    reference_moments[zdm::zebra::Moment::identity, 2, 0, 0, 0] = -1.0/3.0;
    reference_moments[zdm::zebra::Moment::x, 1, 1, 1, 0] = -1.0/sqrt5;
    reference_moments[zdm::zebra::Moment::y, 1, 1, 1, 1] = -1.0/sqrt5;
    reference_moments[zdm::zebra::Moment::z, 1, 1, 0, 0] = 1.0/sqrt5;
    reference_moments[zdm::zebra::Moment::r2, 0, 0, 0, 0] = 1.0/5.0;
    reference_moments[zdm::zebra::Moment::r2, 2, 0, 0, 0] = -1.0/7.0;
    reference_moments[zdm::zebra::Moment::r2, 4, 0, 0, 0] = -2.0/35.0;

    zdm::zebra::RadonMomentArray<double, zdm::zebra::MomentCategory::transverse>
    moments{order + 4};

    zdm::zebra::RadonTransformer<zdm::DistType::aniso, zdm::zebra::MomentCategory::transverse>{order}
        .evaluate_transformed_moments(expansion, moments);

    constexpr double tol = 1.0e-13;

    const std::array moment_labels = {
        zdm::zebra::Moment::identity,
        zdm::zebra::Moment::x,
        zdm::zebra::Moment::y,
        zdm::zebra::Moment::z,
        zdm::zebra::Moment::r2
    };

    bool success = true;
    for (auto label : moment_labels)
    {
        for (std::size_t n : moments[label].indices())
        {
            for (std:: size_t l : moments[label, n].indices())
            {
                for (std::size_t m : moments[label, n, l].indices())
                {
                    success = success
                        && is_close(
                            moments[label, n, l, m, 0],
                            reference_moments[label, n, l, m, 0],
                            tol);
                    success = success
                        && is_close(
                            moments[label, n, l, m, 1],
                            reference_moments[label, n, l, m, 1],
                            tol);
                }
            }
        }
    }

    if (!success)
    {
        for (auto label : moment_labels)
        {
            std::println("{}: moment reference", zdm::zebra::to_string(label));
            for (std::size_t n : moments[label].indices())
            {
                for (std:: size_t l : moments[label, n].indices())
                {
                    for (std::size_t m : moments[label, n, l].indices())
                    {
                        std::println("{}, {}, {}: [{}, {}] [{}, {}]", n, l, m,
                                     moments[label, n, l, m, 0], moments[label, n, l, m, 1],
                                     reference_moments[label, n, l, m, 0], reference_moments[label, n, l, m, 1]);
                    }
                }
            }
        }
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
        recursion_data, expansion, r2_radon);

    zdm::zebra::IsotropicRadonMomentArray<double, zdm::zebra::MomentCategory::transverse>
    composite_moments{order + 4};

    zdm::zebra::detail::transverse_radon_moments(radon, r2_radon, composite_moments);

    zdm::zebra::IsotropicRadonMomentArray<double, zdm::zebra::MomentCategory::transverse>
    direct_moments{order + 4};

    zdm::zebra::RadonTransformer<zdm::DistType::iso, zdm::zebra::MomentCategory::transverse>{order}
        .evaluate_transformed_moments(expansion, direct_moments);

    constexpr double tol = 1.0e-13;

    const std::array moment_labels = {
        zdm::zebra::IsoMoment::identity,
        zdm::zebra::IsoMoment::linear,
        zdm::zebra::IsoMoment::quadratic,
    };

    bool success = true;
    for (auto label : moment_labels)
    {
        for (std::size_t n : direct_moments[label].indices())
            success = success && is_close(direct_moments[label, n], composite_moments[label, n], tol);
    }

    if (!success)
    {
        for (auto label : moment_labels)
        {
            std::println("{}: direct composite", zdm::zebra::to_string(label));
            for (std::size_t n : direct_moments[label].indices())
                std::println("{}: {} {}", n, direct_moments[label, n], composite_moments[label, n]);
        }
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
        recursion_data, expansion, r2_radon);

    zdm::zebra::IsotropicRadonMomentArray<double, zdm::zebra::MomentCategory::transverse>
    composite_moments{order + 4};
    zdm::zebra::detail::transverse_radon_moments(radon, r2_radon, composite_moments);

    zdm::zebra::IsotropicRadonMomentArray<double, zdm::zebra::MomentCategory::transverse>
    direct_moments{order + 4};
    zdm::zebra::RadonTransformer<zdm::DistType::iso, zdm::zebra::MomentCategory::transverse>{order}
        .evaluate_transformed_moments(expansion, direct_moments);

    constexpr double tol = 1.0e-13;

    const std::array moment_labels = {
        zdm::zebra::IsoMoment::identity,
        zdm::zebra::IsoMoment::linear,
        zdm::zebra::IsoMoment::quadratic,
    };

    bool success = true;
    for (auto label : moment_labels)
    {
        for (std::size_t n : direct_moments[label].indices())
            success = success && is_close(direct_moments[label, n], composite_moments[label, n], tol);
    }

    if (!success)
    {
        for (auto label : moment_labels)
        {
            std::println("{}: direct composite", zdm::zebra::to_string(label));
            for (std::size_t n : direct_moments[label].indices())
                std::println("{}: {} {}", n, direct_moments[label, n], composite_moments[label, n]);
        }
    }

    return success;
}

} // namespace

int main()
{
    assert(test_radon_transformer_iso_trans_is_correct_for_constant_distribution(10));
    assert(test_radon_transformer_aniso_trans_is_correct_for_constant_distribution(10));

    assert(test_isotropic_zernike_transverse_radon_helper_components_are_consistent(20, 0));
    assert(test_isotropic_zernike_transverse_radon_helper_components_are_consistent(20, 2));
    assert(test_isotropic_zernike_transverse_radon_helper_components_are_consistent(20, 4));

    assert(test_isotropic_zernike_transverse_radon_helper_components_are_consistent(20));
    assert(test_isotropic_zernike_transverse_radon_helper_components_are_consistent(21));
}
