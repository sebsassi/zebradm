#pragma once

#include <numbers>

#include "linalg.hpp"

#include "coordinates/coordinate_functions.hpp"

namespace velocity
{

template <typename FieldType>
concept bounded_distribution = requires (const FieldType& dist, const Vector<double, 3>& velocity)
{
    { dist(velocity) } -> std::same_as<double>;
    { dist.normalization() } -> std::same_as<double>;
    { dist.max_velocity() } -> std::same_as<double>;
};

/*
Spherically truncated isotropic Gaussian distribution.
*/
class STIG
{
public:
    STIG(double max_velocity, double disp_velocity):
        m_max_velocity(max_velocity), m_disp_velocity(disp_velocity), m_inv_v_disp_sq(1.0/(disp_velocity*disp_velocity)), m_normalization(eval_norm(max_velocity, disp_velocity)) {}

    [[nodiscard]] double max_velocity() const noexcept { return m_max_velocity; }
    [[nodiscard]] double disp_velocity() const noexcept { return m_disp_velocity; }
    [[nodiscard]] double normalization() const noexcept { return m_normalization; }
    double operator()(const std::array<double, 3>& v) const noexcept
    {
        const double v_sq = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
        return m_normalization*std::exp(-v_sq*m_inv_v_disp_sq);
    }
private:
    static double eval_norm(double max_velocity, double disp_velocity) noexcept
    {
        constexpr double exp_pref = 2.0*std::numbers::inv_sqrtpi;
        constexpr double pi_sqrt_cb = std::numbers::pi/std::numbers::inv_sqrtpi;

        const double u_max = max_velocity/disp_velocity;
        const double u_max_sq = u_max*u_max;
        const double v_esc_part = std::erf(-u_max) - exp_pref*u_max*std::exp(-u_max_sq);

        const double v_rot_cb = disp_velocity*disp_velocity*disp_velocity;
        return 1.0/(pi_sqrt_cb*v_rot_cb*v_esc_part);
    }

    double m_max_velocity;
    double m_disp_velocity;
    double m_inv_v_disp_sq;
    double m_normalization;
};

// Standard Halo Model distribution
using SHM = STIG;

template <typename FieldType>
constexpr std::array<FieldType, 3> sorted(const std::array<FieldType, 3>& a)
{
    return {
        std::max(std::max(a[0], a[1]), a[2]),
        std::max(std::min(a[0], a[1]), std::min(std::max(a[0], a[1]), a[2])),
        std::min(std::min(a[0], a[1]), a[2])
    };
}

double cerf(double x)
{
    if (x > 0.01) return std::erf(x)/x;

    constexpr double a0 = 1.0;
    constexpr double a1 = 1.0/3.0;
    constexpr double a2 = 1.0/10.0;
    constexpr double a3 = 1.0/42.0;
    constexpr double a4 = 1.0/216.0;
    constexpr double a5 = 1.0/1320.0;

    const double x_sq = x*x;
    const double poly = a0 - x_sq*(a1 - x_sq*(a2 - x_sq*(a3 - x_sq*(a4 - x_sq*a5))));
    return (2.0*std::numbers::inv_sqrtpi)*poly;
}


/*
Spherically truncated anisotropic Gaussian distribution.
*/
class STAG
{
public:
    STAG(double max_velocity, double disp_velocity, const std::array<double, 3>& sigma):
        m_max_velocity(max_velocity), m_disp_velocity(disp_velocity), m_inv_v_disp_sq(1.0/(disp_velocity*disp_velocity)), m_normalization(eval_norm(max_velocity, disp_velocity, sigma)), m_sigma(sigma),
        m_inv_sigma_sq(std::array<double, 3>{
            1.0/(sigma[0]*sigma[0]), 1.0/(sigma[1]*sigma[1]), 1.0/(sigma[2]*sigma[2])
        }) {}


    [[nodiscard]] double max_velocity() const noexcept { return m_max_velocity; }
    [[nodiscard]] double disp_velocity() const noexcept { return m_disp_velocity; }
    [[nodiscard]] double normalization() const noexcept { return m_normalization; }

    double operator()(const std::array<double, 3>& v) const noexcept
    {
        const std::array<double, 3> u_sq = {
            v[0]*v[0]*m_inv_v_disp_sq, v[1]*v[1]*m_inv_v_disp_sq,
            v[2]*v[2]*m_inv_v_disp_sq
        };
        const double exp_arg = u_sq[0]*m_inv_sigma_sq[0]
                + u_sq[1]*m_inv_sigma_sq[1]
                + u_sq[2]*m_inv_sigma_sq[2];
        return m_normalization*std::exp(-exp_arg);
    }

private:
    double eval_norm(
        double max_velocity, double disp_velocity,
        const std::array<double, 3>& sigma)
    {
        constexpr double two_pi_sqrt = std::numbers::sqrt2/std::numbers::inv_sqrtpi;
        constexpr double two_pi_32 = two_pi_sqrt*two_pi_sqrt*two_pi_sqrt;
        const double u_max = max_velocity/disp_velocity;
        const double v_rot_cb = disp_velocity*disp_velocity*disp_velocity;
        const double det = std::sqrt(sigma[0]*sigma[1]*sigma[2]);
        const std::array<double, 3> sigma_decr = sorted(sigma);
        const double first_term = two_pi_32*det*std::erf(std::sqrt(0.5/sigma_decr[0])*u_max);

        const double second_term = norm_trunc_term(sigma_decr, u_max);
        return first_term - second_term;
    }

    double norm_trunc_term(
        const std::array<double, 3>& sigma_decr, double u_max)
    {
        // Gauss-Legendre nodes and weights
        constexpr std::array<double, 15> nodes = {
            0.5147184255531769583302503e-1, 0.153869913608583546963794675, 0.254636926167889846439804839, 0.352704725530878113471037373, 0.447033769538089176780610284, 0.536624148142019899264169965, 0.620526182989242861140477635, 0.697850494793315796932292385, 0.767777432104826194917977325, 0.829565762382768397442898081, 0.882560535792052681543116509, 0.926200047429274325879324301, 0.960021864968307512216870991, 0.983668123279747209970032601, 0.996893484074649540271630100};
        constexpr std::array<double, 15> weights = {
            0.1028526528935588403412856, 0.1017623897484055045964290, 0.9959342058679526706278018e-1, 0.9636873717464425963946864e-1, 0.9212252223778612871763266e-1, 0.8689978720108297980238752e-1, 0.8075589522942021535469516e-1, 0.7375597473770520626824384e-1, 0.6597422988218049512812820e-1, 0.5749315621761906648172152e-1, 0.4840267283059405290293858e-1, 0.3879919256962704959680230e-1, 0.2878470788332336934971862e-1, 0.1846646831109095914230276e-1, 0.7968192496166605615469690e-2};

        const double a = u_max/sigma_decr[0];
        const double b = u_max/sigma_decr[1];
        const double c = u_max/sigma_decr[2];
        const double a2 = a*a;
        const double b2 = b*b;
        const double c2 = c*c;

        double res = 0.0;
        for (std::size_t i = 0; i < nodes.size(); ++i)
        {
            double xp = 0.25*std::numbers::pi*(1.0 + nodes[i]);
            double xm = 0.25*std::numbers::pi*(1.0 - nodes[i]);
            res += weights[i]*(trunc_term_integrand(xp, a2, b2, c2) + trunc_term_integrand(xm, a2, b2, c2));
        }
        return std::numbers::pi*res;
    }

    double trunc_term_integrand(double phi, double a2, double b2, double c2)
    {
        const double ellipse = 0.5*((b2 + c2) + (b2 - c2)*std::cos(2*phi));
        const double cerf_arg = std::sqrt(a2 - ellipse);
        return std::exp(-0.5*ellipse)*cerf(cerf_arg);
    }

    double m_max_velocity;
    double m_disp_velocity;
    double m_inv_v_disp_sq;
    double m_normalization;
    std::array<double, 3> m_sigma;
    std::array<double, 3> m_inv_sigma_sq;
};

}