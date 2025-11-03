Celestial Coordinate Systems
============================

The motion of our detector laboratory with respect to the dark matter Halo is of fundamental
importance in direct detection experiments. This motion gives rise to both annual and daily
variation in the detection rate of dark matter experiments, and of course also determines the
directionality of the dark matter signal in experiments with directional capabilities. Therefore,
we need a set of well-defined coordinate transforms to connect not only the velocity, but also
the orientation of our detector with that of the dark matter halo.

This library implements a set of coordinate transforms between standard reference coordinate
systems, and the laboratory frame. These follow the IERS Conventions [IERS2010]_ where
applicable.

.. note::
   All coordinate systems used by this library are right-handed by convention.

The celestial coordinate transformations in this library are types which fulfill the requirements
of the :cpp:concept:`zdm::celestial::parametric_rigid_transform` concept. These are types, which
implement a call operator :cpp:`operator()`, which takes in a single parameter (time), and
typically returns a :cpp:type:`zdm::la::RigidTransform` consisting of a rotation and a translation
(velocity boost). If the transformation is only a rotation, the call operator may also return a
:cpp:type:`zdm::la::RotationMatrix`, or likewise, if it is only a translation, it may return a
:cpp:type:`zdm::la::Vector` corresponding to the translation vector.

For instance if we wanted the average velocity of dark matter in a particular Horizontal Coordinate
System (HCS), we could use the class :cpp:type:`zdm::celestial::GCStoHCS` to transform the average
velocity of dark matter in the Galactic Coordinate System (GCS), which happens to be zero, to our
HCS.

.. code:: cpp

    const double longitude = std::numbers::pi/2.0;
    const double latitude = std::numbers::pi/4.0;
    const double circular_speed = 233.0;

    auto gcs_to_hcs = zdm::celestial::GCStoHCS(longitude, latitude, circular_velocity);

    const auto utc_date = Time{
        .year = 2025, .mon = 12, .mday = 19, .hour = 13, .min = 45, .sec = 34, .msec = 325
    };
    const double days_since_j2000 = ut1_from_time<zdm::time::j2000_utc>(utc_date);
    const zdm::la::RigidTransform<double, 3> gcs_to_hcs_of_date = gcs_to_hcs(days_since_j2000);

    const zdm::la::Vector<double, 3> dm_velocity_lab = gcs_to_hcs_of_date(zdm::la::Vector<double, 3>{});

The main reference system which connects many of the coordinate systems used in this library is the
International Celestial Reference System (ICRS). This is a coordinate system with its origin at the
solar system barycenter, and which is defined to be nonrotating with respect to extragalactic
radio sources. It is defined in such a way that on the J2000 epoch its orientation approximately
matches that of the Equatorial Coordinate System (ECS) [1]_. That is, its equator roughly matches
the J2000 equator, and its x-axis the March equinox of J2000. The direction of the z-axis of the
ICRS is called the North Celestial Pole (NCP).

In the framework of this library, the local velocity distribution of dark matter is typically
assumed to be defined in the Galactic Coordinate System (GCS). It is a coordinate system, which for
a point in the galactic disk (e.g. the solar system) is at any instance centered on that point, but
has no overall motion with respect to the Galactic Core (GC). This is a coordinate system where the
mean velocity of dark matter is zero (under the standard assumptions that the dark matter halo
doesn't have significant angular momentum, and that there are no local streams).

The orientation of the GCS is determined such that its z-axis points in the direction of the North
Galactic Pole (NGP), defined as the direction perpendicular to the galactic equatorial plane. Its
x-axis is in the direction, where the great circle going through the NGP and GC crosses the
equatorial plane.

The velocity of the ICRS in the GCS is given by the sum of the local circular velocity and peculiar
velocity of the solar system

.. math::

    \vec{v}_{\text{GCS}\rightarrow\text{ICRS}} = \vec{v}_\text{circ} + \vec{v}_\text{pec}.

For the peculiar velocity this library supplies the constant `peculiar_velocity_sbd_2010` per the
values of [SBD2010]_.

The orientation of the GCS in the ICRS can be expressed in terms of the ICRS right ascension and
declination of the NGP, and the GCS longitude of the NCP as [2]_

.. math::

    R_{\text{GCS}\rightarrow\text{ICRS}}
    = R_Z(\pi - \alpha_\text{NGP})R_Y(\pi/2 - \delta_\text{NGP})R_Z(l_\text{NCP}).

The transformation from the GCS to ICRS is implemented by the class
:cpp:type:`zdm::celestial::GCStoICRS`, although there is typically no need to deal with it directly
because one more often wants to use composite transforms like :cpp:type:`zdm::celestial::GCStoHCS`
to transform to a more common laboratory frame.

.. code:: cpp

    auto gcs_to_icrs = zdm::celestial::GCStoICRS{};
    zdm::la::RigidTransform<double, 3> gcs_to_icrs_j2000 = gcs_to_icrs(0.0);
    zdm::la::Vector<double, 3> dm_velocity_icrs = gcs_to_icrs_j2000(zdm::la::Vector<double, 3>{});

Although the GCS to ICRS transformation is indpendent of time, all coordinate transforms in the
library are parametrized by time, expressed in days since the J2000 epoch. Calling the coordinate
transform with a time parameter gives the value of the transform at that instance.

The orbital motion of the Earth is implemented in the transformation from the ICRS to the
Geocentric Celestial Reference System (GCRS). The GCRS has the same orientation as the ICRS, but
where the ICRS is centered at the barycenter of the solar system, the GCRS is geocentric. Their
velocity difference is therefore given by the orbital velocity of Earth (or viewed from Earth, the
velocity of the solar system barycenter). The class :cpp:type:`zdm::celestial::ICRStoGCRS` returns
this velocity difference

.. code:: cpp

    auto icrs_to_gcrs = zdm::celestial::ICRStoGCRS{};
    zdm::la::Vector<double, 3> ssbr_velocity = icrs_to_gcrs(0.0);

For completeness we may note that the ICRS to GCRS transformation relies on another transformation
:cpp:type:`zdm::celestial::ECStoICRS` to express the orbital velocity origincally expressed in the
Ecliptic Coordinate System (ECS) of J2000 in the ICRS.

The primary terrestrial reference system is the International Terrestrial Reference System (ITRS).
This is a geocentric coordinate system whose z-axis matches Earth's rotational axis, and whose
x-axis is on the prime meridian. For technical reasons, it is connected to the GCRS via a
three-stage transform consinsting of three rotations. Earth's axis of rotation moves with respect
to the ICRS pole, and this rotation is separated into precession and nutation due to gravitational
perturbations, and polar motion due to terrestrial effects. For this purpose, the Celestial
Intermediate Pole (CIP) has been defined, whose motion includes the precession and nutation, but
not the polar motion. This leads to the definition of the Celestial Intermediate Reference System
(CIRS), which has the CIP as its pole, but does not rotate with Earth, as well as the Terrestrial
Intermediate reference System (TIRS), which has the CIP as its pole and does rotate with Earth.
This leads to the transformation sequence

.. math::

   \text{GCRS} \rightarrow \text{CIRS} \rightarrow \text{TIRS} \rightarrow \text{ITRS}.

The transformation from GCRS to CIRS adds the precession and nutation of the CIP, the
transformation from CIRS to TIRS rotates the coordinates with the Earth Rotation Angle (ERA)

.. math::

   R_{\text{CIRS}\rightarrow\text{TIRS}} = R_Z(\text{ERA})

.. math::

   \text{ERA} = 2\pi(0.7790572732640 + 1.00273781191135448\text{UT1}_\text{J2000}).

Here :math:`\text{UT1}_\text{J2000}` denotes the UT1 time (in days) since the J2000 epoch.

The GCRS to CIRS and CIRS to TIRS transformations are implemented by the respective classes
:cpp:type:`zdm::celestialGCRStoCIRS` and :cpp:type:`zdm::celestial::CIRStoTIRS`. The transformation
between TIRS and ITRS is close enough to identity that it is not implemented in this library.
While the class :cpp:type:`zdm::celestial::TIRStoITRS` exists, it does not implement a useable
transformation.

Reference
---------

Types
^^^^^

.. doxygenclass:: zdm::celestial::Composite
    :project: zebradm
    :members:

.. doxygenclass:: zdm::celestial::Inverse
    :project: zebradm
    :members:

.. doxygenclass:: zdm::celestial::GCStoICRS
    :project: zebradm
    :members:

.. doxygenclass:: zdm::celestial::ECStoICRS
    :project: zebradm
    :members:

.. doxygenclass:: zdm::celestial::ICRStoGCRS
    :project: zebradm
    :members:

.. doxygenclass:: zdm::celestial::GCRStoCIRS
    :project: zebradm
    :members:

.. doxygenclass:: zdm::celestial::CIRStoTIRS
    :project: zebradm
    :members:

.. doxygenclass:: zdm::celestial::TIRStoITRS
    :project: zebradm
    :members:

.. doxygenclass:: zdm::celestial::ITRStoHCS
    :project: zebradm
    :members:

.. doxygenclass:: zdm::celestial::GCStoHCS
    :project: zebradm
    :members:

.. doxygenclass:: zdm::celestial::GCStoCIRS
    :project: zebradm
    :members:

.. doxygenclass:: zdm::celestial::CIRStoHCS
    :project: zebradm
    :members:

.. doxygenclass:: zdm::celestial::TIRStoHCS
    :project: zebradm
    :members:

Footnotes
---------

.. [1] The J2000 epoch is defined as noon January 1st, 2000 Terrestrial Time, which corresponds to
    11:58:55.816 January 1st, 2000 UTC.

.. [2] Rotation matrices in this documentation use express passive transformations with positive
    angles corresponding tcorresponding to counterclockwise rotations.

References
----------

.. [IERS2010] IERS Conventions (2010). Gérard Petit and Brian Luzum (eds.). (IERS Technical Note ;
    36) Frankfurt am Main: Verlag des Bundesamts für Kartographie und Geodäsie, 2010. 179 pp., ISBN
    3-89888-989-6

.. [SBD2010] Schönrich, R., Binney, J., and Dehnen, W., “Local kinematics and the local standard of
    rest”, Monthly Notices of the Royal Astronomical Society, vol. 403, no. 4, OUP, pp. 1829–1833,
    2010. doi:10.1111/j.1365-2966.2010.16253.x.
