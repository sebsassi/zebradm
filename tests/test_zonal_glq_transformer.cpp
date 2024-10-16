#include "zonal_glq_transformer.hpp"

constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol;
}

template <typename GridLayout, zest::st::SHNorm NORM>
bool test_zonal_glq_forward_transform_expands_Y00()
{
    constexpr std::size_t order = 6;

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double colat)
    {
        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm;
    };
    
    zebra::ZonalGLQTransformer<NORM, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;
    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t l = 0; l < expansion.size(); ++l)
    {
        if (l == 0)
        {
            if (is_close(expansion[l], reference_coeff, tol))
                success = success && true;
            else
                success = success && false;
        }
        else
        {
            if (is_close(expansion[l], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
    }

    if (!success)
    {
        for (std::size_t l = 0; l < order; ++l)
                std::printf("%lu %f\n", l, expansion[l]);
    }
    return success;
}

template <typename GridLayout, zest::st::SHNorm NORM>
bool test_zonal_glq_forward_transform_expands_Y10()
{
    constexpr std::size_t order = 6;

    auto function = []([[maybe_unused]] double lon, double colat)
    {
        const double z = std::cos(colat);

        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm*std::sqrt(3.0)*z;
    };
    
    zebra::ZonalGLQTransformer<NORM, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;
    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t l = 0; l < expansion.size(); ++l)
    {
        if (l == 1)
        {
            if (is_close(expansion[l], reference_coeff, tol))
                success = success && true;
            else
                success = success && false;
        }
        else
        {
            if (is_close(expansion[l], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
    }

    if (!success)
    {
        for (std::size_t l = 0; l < order; ++l)
                std::printf("%lu %f\n", l, expansion[l]);
    }
    return success;
}

template <typename GridLayout, zest::st::SHNorm NORM>
bool test_zonal_glq_forward_transform_expands_Y20()
{
    constexpr std::size_t order = 6;

    auto function = []([[maybe_unused]] double lon, double colat)
    {
        const double z = std::cos(colat);

        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm*0.5*std::sqrt(5.0)*(3.0*z*z - 1.0);
    };
    
    zebra::ZonalGLQTransformer<NORM, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;
    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t l = 0; l < expansion.size(); ++l)
    {
        if (l == 2)
        {
            if (is_close(expansion[l], reference_coeff, tol))
                success = success && true;
            else
                success = success && false;
        }
        else
        {
            if (is_close(expansion[l], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
    }

    if (!success)
    {
        for (std::size_t l = 0; l < order; ++l)
                std::printf("%lu %f\n", l, expansion[l]);
    }
    return success;
}

template <typename GridLayout, zest::st::SHNorm NORM>
bool test_zonal_glq_forward_transform_expands_Y21()
{
    constexpr std::size_t order = 6;

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
    };
    
    zebra::ZonalGLQTransformer<NORM, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t l = 0; l < expansion.size(); ++l)
    {
        if (is_close(expansion[l], 0.0, tol))
            success = success && true;
        else
            success = success && false;
    }

    if (!success)
    {
        for (std::size_t l = 0; l < order; ++l)
                std::printf("%lu %f\n", l, expansion[l]);
    }
    return success;
}

template <typename GridLayout, zest::st::SHNorm NORM>
bool test_zonal_glq_forward_transform_expands_Y30()
{
    constexpr std::size_t order = 6;

    auto function = []([[maybe_unused]] double lon, double colat)
    {
        const double z = std::cos(colat);

        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm*0.5*std::sqrt(7.0)*(5.0*z*z - 3.0)*z;
    };
    
    zebra::ZonalGLQTransformer<NORM, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;
    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t l = 0; l < expansion.size(); ++l)
    {
        if (l == 3)
        {
            if (is_close(expansion[l], reference_coeff, tol))
                success = success && true;
            else
                success = success && false;
        }
        else
        {
            if (is_close(expansion[l], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
    }

    if (!success)
    {
        for (std::size_t l = 0; l < order; ++l)
                std::printf("%lu %f\n", l, expansion[l]);
    }
    return success;
}

template <typename GridLayout, zest::st::SHNorm NORM>
bool test_zonal_glq_forward_transform_expands_Y31()
{
    constexpr std::size_t order = 6;

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    };
    
    zebra::ZonalGLQTransformer<NORM, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t l = 0; l < expansion.size(); ++l)
    {
        if (is_close(expansion[l], 0.0, tol))
            success = success && true;
        else
            success = success && false;
    }

    if (!success)
    {
        for (std::size_t l = 0; l < order; ++l)
                std::printf("%lu %f\n", l, expansion[l]);
    }
    return success;
}

template <typename GridLayout, zest::st::SHNorm NORM>
bool test_zonal_glq_forward_transform_expands_Y4m3()
{
    constexpr std::size_t order = 6;

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm*std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon);
    };
    
    zebra::ZonalGLQTransformer<NORM, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t l = 0; l < expansion.size(); ++l)
    {
        if (is_close(expansion[l], 0.0, tol))
            success = success && true;
        else
            success = success && false;
    }

    if (!success)
    {
        for (std::size_t l = 0; l < order; ++l)
                std::printf("%lu %f\n", l, expansion[l]);
    }
    return success;
}


template <typename GridLayout, zest::st::SHNorm NORM>
void test_glq()
{
    assert((test_zonal_glq_forward_transform_expands_Y00<GridLayout, NORM>()));
    assert((test_zonal_glq_forward_transform_expands_Y10<GridLayout, NORM>()));
    assert((test_zonal_glq_forward_transform_expands_Y20<GridLayout, NORM>()));
    assert((test_zonal_glq_forward_transform_expands_Y21<GridLayout, NORM>()));
    assert((test_zonal_glq_forward_transform_expands_Y30<GridLayout, NORM>()));
    assert((test_zonal_glq_forward_transform_expands_Y31<GridLayout, NORM>()));
    assert((test_zonal_glq_forward_transform_expands_Y4m3<GridLayout, NORM>()));
}

int main()
{
    test_glq<zest::st::LatLonLayout<>, zest::st::SHNorm::GEO>();
    test_glq<zest::st::LonLatLayout<>, zest::st::SHNorm::GEO>();
    test_glq<zest::st::LatLonLayout<>, zest::st::SHNorm::QM>();
    test_glq<zest::st::LonLatLayout<>, zest::st::SHNorm::QM>();
}