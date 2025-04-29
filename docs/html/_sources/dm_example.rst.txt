Example: DM--nucleon scattering
===============================

In the nonrelativistic effective theory framework of nuclear scattering, the direct detection
scattering rate for dark matter scattering off nuclear targets can be expressed in the form

.. math::

    \frac{d^2R_S}{dEd\Omega} = \frac{1}{64\pi^2}\frac{\rho_0}{m_\text{DM}^3m_\text{N}^2}(
        F(q^2)\mathcal{R}[f](\hat{q},v_\text{min}) + F_\perp(q^2)\mathcal{R}_\perp[f](\hat{q},v_\text{min})).

Here :math:`\rho_0` is the local dark matter density, :math:`m_\text{DM}` is the mass of the dark
matter particle, :math:`m_\text{N}` is the mass of the target nucleus, :math:`\vec{q}` is the
momentum transfer to the nucleus, and

.. math::

    v_\text{min} = \frac{q}{2\mu_\text{DM,N}},

with :math:`\mu_\text{DM,N}` is the reduced mass of the DM--nucleus system. See the article
`arxiv:2504.19714 <https://arxiv.org/abs/2504.19714>`_ for a detailed discussion.

The functions :math:`F(q^2)` and :math:`F_\perp(q^2)` depend on the effective theory couplings of
dark matter to nucleons, and on the nuclear response functions. We do not concern ourselves with
their details here, however, but it is worth noting that in the limit of small momentum transfer
they can be regarded as polynomials in :math:`q^2`.

The functions :math:`\mathcal{R}[f]` and :math:`\mathcal{R}_\perp[f]` denote the Radon and
transverse Radon transforms of the dark matter velocity distribution :math:`f` in the laboratory
frame, as defined in the theory section.

This example demonstrates how this library could be used to calculate the energy-differential event rate

.. math::

    \frac{dR}{dE} = \int S(\vec{q})\frac{d^2R_S}{dEd\Omega}\,d\Omega.

Disclaimer
----------

There are a number of input quantities whose origins we need to consider for the full calculation.
Apart from the parameters discussed above, there is also the laboratory velocity relative to the
dark matter distribution, :math:`\vec{v}_\text{lab}`, which appears in the formula for the lab
frame velocity distribution :math:`f(\vec{v} + \vec{v}_\text{lab})`.

Furthermore, there is a hidden parameter, which is the relative orientation of the coordinate
systems in which the distribution :math:`f(\vec{v})` is defined, and the coordinate system of the
response :math:`S(\vec{q})`. Conventionally, the former is in galactic coordinates, while the
latter is in lab coordinates, and this needs to be dealt with.

This library is primarily focused on the computationally challenging part of evaluating the Radon
transforms in a timely manner, and does not provide general facilities for evaluation of the other
input parameters. Therefore, this example is about what form the inputs will need to take so that
they can be used with this library. Parts that are outside the scope of this library are assumed as
given.

Inputs
------

We will assume that there exists a number of functions that evaluate the input parameters we need
to compute the angle-integrated Radon transforms, and it is up to the user of the lbrary how those
functions are implemented in practice. These magic functions are as follows

.. code:: cpp

    std::vector<double>
    time_interval(std::string_view start, std::string_view end, std::size_t count);

    std::vecto<std::array<double, 3>>
    compute_lab_velocities_equatorial(std::span<double> times);
    
    std::vector<double> compute_earth_rotation_angles(std::span<double> times);
    zdm::Matrix<double, 3, 3> rotation_matrix_from_equatorial_to_galactic();
    std::array<double, 3> euler_angles_from_lab_to_polar(double lon, double lat);

    std::vector<std::array<double, 2>>
    compute_eft_responses(std::span<double> momentum_transfers);
    
    zdm::SHExpansionVector
    get_detector_response(
        std::span<double> momentum_transfers, std::size_t resp_order);

    DistributionParams get_distribution_params();
    
    double velocity_distribution(
        const std::array<double, 3>& velocity, const DistributionParams& params);

This is approximately the scope of things we'd need to implement to get to the point where we can
evaluate the angle-integrated Radon transform. Some of the the things here have been simplified so
that we don't get bogged down in irrelevant details. For example, in reality the function
``compute_eft_responses`` would also take as parameters the EFT coefficients, parameters of the
target nucleus, and so on. But in this example, we do not care so much about the implementations or
inputs to these functions, just their outputs.

There is already one important aspect that can be observed here. For performance reasons, the
library requires the velocity distribution and response to have coordinate systems whose z-axes are
aligned, but which may differ by an angle in the xy-plane. This is no problem, because the Earth's
axis of rotation does not change, to a good approximation, on time scales at which direct detection
experiments operate. Therefore, aligning the z-axes of the coordinates of the velocity distribution
only requires a time-independent rotation on both. The velocity distribution is rotated to the
equatorial coordinate system (technically, the geocentric celestial refernce system), and the
response is rotated to a horizontal coordinate system at the north pole (which I call "polar"
coordinate system here for brevity, and which technically is the international terrestrial reference
system). These two coordinate systems then differ by the Earth rotation angle.

The velocity distribution here is assumed to have a parametric formula, but the response is defined
a bit more ambiguously, because in reality it would likely be generated from some numerical data,
so we just have a function, which gives a collection of spherical harmonic expansions given a
collection of momentum transfer magnitudes.

To avoid having to specify a definition of time, its origin and units, I have defined a function,
which just gives us a number of times given a start date, end date, and a count

.. code:: cpp

    std::vector<double> times = time_interval("2000-01-01", "2001-01-02", 24);

Doesn't matter what units these are in, because we will just use them to compute things we actually
care about

.. code:: cpp

    std::vector<std::array<double, 3>> v_lab_eq = lab_velocities_equatorial(times);
    std::vector<double> era = earth_rotation_angles(times);

Then we have the static coordinate transforms

.. code:: cpp

    zdm::Matrix<double, 3, 3> equ_to_gal = rotation_matrix_from_equatorial_to_galactic();

    constexpr double lon = 0.5*std::numbers::pi;
    constexpr double lat = 0.25*std::numbers::pi;
    std::array<double, 3> lab_to_polar = euler_angles_from_lab_to_polar(lon, lat);

Apart from a very conveniently located detector site, it's notable that we define one of these
rotations in terms of a rotation matrix, and the other in terms of Euler angles. The reason for
that is that we are going to apply them under different circumstances.

Next we wish to generate a collection of energies at which the energy differential event rate will
be evaluated. However, something we need to take into account is that the energy is bound by the
inequality

.. math::

    v_\text{min} \leq v_\text{lab} + v_\text{esc},

where :math:`v_\text{esc}` is the escape velocity. When this inequality does not hold, the event
rate is zero, so there is no point computing the event rate outside this range. We could just
generate the :math:`v_\text{min}` values directly, but that may not be desirable if we want the
energies to be equispaced, since :math:`v_\text{min}` is not a linear function of energy

.. math::

    v_\text{min} = \sqrt{\frac{m_\text{N}E}{2\mu_\text{DM,N}}}.

However, this gives a straightforward upperbound for the energy

.. math::

    E \leq \frac{2\mu_\text{DM,N}}{m_\text{N}}(v_\text{lab} + v_\text{esc})^2.

This is straightforward enough to implement here

.. code:: cpp

    std::vector<double> generate_energies(
        double reduced_mass, double nuclear_mass, double v_lab_max, double v_esc,
        std::size_t count)
    {
        const double v_minmax = v_lab_max + v_esc;
        const double emax
            = std::sqrt(2.0*reduced_mass/nuclear_mass)*v_minmax*v_minmax;

        std::vector<double> energies(count);
        for (std::size_t i = 0; i < count; ++i)
            energies[i] = emax*double(i)/double(count - 1);

        return energies;
    }

    std::vector<double> vmin_from(
        std::span<double> energies, double reduced_mass, double nuclear_mass)
    {
        const double prefactor = nuclear_mass/(2.0*reduced_mass);

        std::vector<double> vmin(energies.size());
        for (std::size_t i = 0; i < count; ++i)
            vmin[i] = std::sqrt(prefactor*energies[i]); 

        return vmin;
    }

    std::vector<double> momentum_transfers_from(
        std::span<double> energies, double nuclear mass)
    {
        const double prefactor = 2.0*nuclear_mass;

        std::vector<double> momentum_transfers(energies.size());
        for (std::size_t i = 0; i < count; ++i)
            momentum_transfers[i] = std::sqrt(prefactor*energies[i]); 

        return momentum_transfers;
    }

The value of ``v_lab_max`` we need to calculate from the list of lab velocities we generated above

.. code:: cpp

    double maximum_lab_velocity(std::span<std::vector<double, 3>> lab_velocities)
    {
        double v_lab_sq_max = 0.0;
        for (const auto& v_lab : lab_velocities)
        {
            const double v_sq = zdm::dot(v_lab, v_lab);
            v_lab_sq_max = std::max(v_lab_sq_max, v_sq);
        }

        return std::sqrt(v_lab_sq_max);
    }

Now we can generate the :math:`v_\text{min}` and momentum transfer values we are after

.. code:: cpp

    std::vector<double> energies = generate_energies(
        reduced_mass, nuclear_mass, maximum_lab_velocity(v_lab_eq), v_esc, 50);
    std::vector<double> v_min = vmin_from(energies, reduced_mass, nuclear_mass);
    std::vector<double> momentum_transfers
        = momentum_transfer_from(energies, nuclear_mass);

Distribution and response
-------------------------

With the momentum transfers, we can get the detector response

.. code:: cpp

    zdm::SHExpansionVector resp = get_detector_response(momentum_transfers);

Now, this response is defined in the lab frame, but we want it in the polar frame, so we need to
rotate it. For this we can use the class :cpp:type:`zest::Rotor`, which enables rotations of
spherical harmonic expansions

.. code:: cpp

    zest::WignerdPiHalfCollection wigner(std::max(resp_order, dist_order));
    zest::Rotor rotor(std::max(resp_order, disp_order));
    for (std::size_t i = 0; resp.extent(); ++i)
        rotor.rotate(resp[i], wigner, lab_to_polar, zest::RotationType::coordinate);

When it comes to the velocity distribution, we need it to have a specific function signature, which
only takes as arguments the spherical coordinates of the velocity itself. The easiest way to do this
is to wrap it in a lambda

.. code:: cpp

    const zdm::Matrix<double, 3, 3> rot_equ_to_gal
        = rotation_matrix_from_equatorial_to_galactic();
    const DistributionParams params = get_distribution_params();
    auto wrapped_distribution = [&](double lat, double colat, double r)
    {
        const std::array<double, 3> v_equ
            = zdm::coordinates::spherical_to_cartesian_phys(lat, colat, r);
        const std::array<double, 3> v_gal = zdm::matmul(rot_equ_to_gal, v_equ);
        return velocity_distribution(v_gal, params);
    };

We can then take the Zernike transform of the wrapped distribution

.. code:: cpp

    zest::ZernikeTransformerNormalGeo zernike_transformer{};
    zdm::ZernikeExpansion dist
        = zernike_transformer.transform(wrapped_distribution, v_esc, dist_order);

Giving ``v_esc`` as the second paramter here essentially tells the transformer that the velocity
distribution is zero for velocities greater than the escape velocity, so that it can internally
scale the coordinates to the unit sphere.

Angle-integrated Radon transform
--------------------------------

We for the most general dark matter event rate, we need both the nontransverse and transverse Radon
transforms, so we choose :cpp:type:`zdm::zebra::AnisotropicTransverseAngleIntegrator`

.. code:: cpp

    zdm::zebra::AnisotropicAngleIntegrator integrator(dist_order, resp_order);
    
Before we go and compute the angle-integrated Radon transforms, there is one very important thing
to account for. The integral that our ``integrator`` computes is defined on the unit ball in the
velocity space. In other words, it is computed in a system of units where :math:`v_\text{esc} = 1`
by definition. Therefore, we need to scale all our units appropriately. In terms of the input,
this means dividing both :math:`v_\text{lab}` and :math:`v_\text{min}` by :math:`v_\text{esc}`

.. code:: cpp

    const double inv_v_esc = 1.0/v_esc;
    
    std::vector<std::array<double, 3>> u_lab_eq = v_lab_eq;
    for (auto& element : u_lab_eq)
        element = zdm::mul(inv_v_esc, element);
    
    std::vector<double> u_min = v_min;
    for (auto& element : v_min)
        element *= inv_v_esc;
    
Then these are the inputs to the integrator

.. code:: cpp

    zest::MDArray<std::array<double, 2>, 2> out({v_lab_eq.size(), v_min.size()});
    integrator.integrate(dist, resp, u_lab_eq, era, u_min, out);

Again, we need to account for the units in which the velocity integral was computed. This means
multiplying the nontransverse Radon transform by :math:`v_\text{esc}^2` and the transverse Radon
transform by :math:`v_\text{esc}^4`

.. code:: cpp

    const double v_esc_2 = v_esc*v_esc;
    const double v_esc_4 = v_esc_2*v_esc_2;
    for (auto& element : out.flatten())
    {
        element[0] *= v_esc_2;
        element[1] *= v_esc_4;
    }

Getting the event rates out
---------------------------

After this, the output is in our original velocity units. After this it is just a matter of
multiplying by the EFT responses, adding the results together, and multiplying by the common
prefactor.

.. code:: cpp

    std::vector<double> eft_responses = compute_eft_responses(momentum_transfers);
    const double prefactor = event_rate_prefactor(dm_density, dm_mass, nuclear_mass);

    zdm::MDArray<double, 2> event_rates(out.extents());
    for (std::size_t i = 0; out.extent(0); ++i)
    {
        for (std::size_t j = 0; out.extent(1); ++j)
            event_rates(i,j) += prefactor*zdm::dot(eft_responses[j], out(i,j));
    }

The function ``event_rate_prefactor`` here is a stand in for the prefactor

.. math::

   \frac{1}{64\pi^2}\frac{\rho_0}{m_\text{DM}^3m_\text{N}^2}.


