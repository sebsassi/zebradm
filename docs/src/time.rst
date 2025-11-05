Time Utilities
==============

The ``time.hpp`` header provides a couple of basic utilities for specifying times to be used in
dark matter event rate calculations. Although the precision requirements of the library do not
require a thorough consideration of the defition of time, this section begins with a brief overview
of different time keeping standards to give the necessary context for time keeping in this library.

This library mainly operates in Universal Time, specifically, UT1. This is a time standard based on
Earth's rotation. It is therefore not strictly linear, as its progression fluctuates due to changes
in Earth's rotation. UT1 is closely related to Coordinated Universal Time (UTC), the main standard
for civilian time keeping. Namely, UTC has been defined to be within 0.9 seconds of UT1, and this
relationship has been maintained with occasional introduction of leap seconds. The difference of
UT1 and UTC is denoted DUT1.

This library does not seek sub-second precision for time keeping. Therefore, because there is no
long term numerical model of DUT1,[1]_ this library regards UTC and UT1 as interchangeable.

The conventional standard for astronomical time keeping is Terrestrial Time (TT), which is related
to International Atomic Time (TAI). Namely, it is defined to be exactly 32.184 seconds ahead of
TAI. These time standards are related to UTC in that at the start of 1972 UTC was set to be 10
seconds behind TAI. Therefore

.. math::

    \text{TT} = \text{UT1} + 42.184\text{s} + \text{leap seconds} \pm 1\text{s}.

Time is always measured relative to some reference point, an *epoch*, which is the moment in time
at which our time variable is zero. This becomes pertinent when we need to calculate values for
time-dependent quantities, and when we have clocks running at different rates. In astronomy, the
modern, conventional reference epoch is at noon on January 1st, 2000, Terrestrial Time, know as
J2000. In UTC, due to the above mentined offset, this translates to 11:58:55.816 on the same date,
given that there are 22 leap seconds before the year 2000.

Setting the zero point of both TT and UT1 at J2000 makes the constant offset go away, and thus

.. math::

    \text{TT}_\text{J2000} = \text{UT1}_\text{J2000} + \text{leap seconds} \pm 1\text{s}.

There have been exactly 5 leap seconds since J2000 at the writing of this documentation. Since leap
seconds are unpredictable and cannot be predicted from a numerical model, and they only lead to a
small offset, this library does not account for them, and effectively assumes

.. math::

    \text{TT}_\text{J2000} \approx \text{UT1}.

Dates in the library are expressible using the :cpp:struct:`zdm::time::Time` type. The exact UTC time
of J2000 is provided as a constant :cpp:var:`zdm::time::j2000_utc`

.. code:: cpp

    constexpr Time j2000_utc = {
        .year = 2000,
        .mon = 1,
        .mday = 1,
        .hour = 11,
        .min = 58,
        .sec = 55,
        .msec = 816
    };

The number of UT1 days that have elapsed since J2000 can be obtained with the function

.. doxygenfunction:: zdm::time::ut1_from_utc
    :project: zebradm
    :no-link:
    :outline:

.. code:: cpp

    const zdm::time::Time random_time = {2025, 11, 5, 12, 10, 51, 342};
    const double ut1 = ut1_from_utc<zdm::time::j2000_utc>(random_time);

Although more often we probably want to generate a number of time points between some start time
and end time. For this purpose there is the :cpp:function:`ut1_interval` function. This function
provides two variants

.. doxygenfunction:: zdm::time::ut1_interval(std::span<double>, Time, Time)
    :project: zebradm
    :no-link:
    :outline:

.. doxygenfunction:: zdm::time::ut1_interval(Time, Time, std::size_t)
    :project: zebradm
    :no-link:
    :outline:

One can be used to produce the UT1 values in place

.. code:: cpp

    const zdm::time::Time start = {2025, 11, 5};
    const zdm::time::Time end = {2025, 11, 6};
    std::vector<double> ut1(24);
    zdm::time::ut1_interval<zdm::time::j2000_utc>(ut1, start, end);

This is useful in performance critical applications where we wish to avoid additional heap
allocations. The other one outputs a :cpp:class:`std::vector` outright

.. code:: cpp

    const zdm::time::Time start = {2025, 11, 5};
    const zdm::time::Time end = {2025, 11, 6};
    std::vector<double> ut1 = ut1_interval<zdm::time::j2000_utc>(start, end, 24);

This library also provides some facilities for dealing with C-style date strings. Namely, the
function :cpp:function:`parse_time` allows parsing date strings into :cpp:struct:`zdm::time::Time`

.. doxygenfunction:: zdm::time::parse_time
    :project: zebradm
    :no-link:
    :outline:

.. code:: cpp

    const zdm::time::Time time = zdm::time::parse_time("November 5, 2025", "%b %d, %Y");

The format supports a subset of conventional format specifiers, which are listed in the
documentation of :cpp:function:`parse_time` below. There are variants of the function
:cpp:function:`ut1_interval`, which take date strings and a format string instead of a
:cpp:struct:`zdm::time::Time`.

Reference
---------

Enums
^^^^^

.. doxygenenum:: zdm::time::DateParseStatus
   :project: zebradm

Types
^^^^^

.. doxygenclass:: zdm::time::TimeZoneOffset
    :project: zebradm
    :members:

.. doxygenclass:: zdm::time::Time
    :project: zebradm
    :members:

Functions
^^^^^^^^^

.. doxygenfunction:: zdm::time::is_leap_year
    :project: zebradm

.. doxygenfunction:: zdm::time::days_in_year
    :project: zebradm

.. doxygenfunction:: zdm::time::day_of_year(std::int32_t, std::uint32_t)
    :project: zebradm

.. doxygenfunction:: zdm::time::day_of_year(std::int32_t, std::uint32_t, std::uint32_t)
    :project: zebradm

.. doxygenfunction:: zdm::time::month_of_year
    :project: zebradm

.. doxygenfunction:: zdm::time::days_until
    :project: zebradm

.. doxygenfunction:: zdm::time::milliseconds_since_epoch
    :project: zebradm

.. doxygenfunction:: zdm::time::parse_time
    :project: zebradm

.. doxygenfunction:: zdm::time::ut1_from_utc
    :project: zebradm

.. doxygenfunction:: zdm::time::ut1_from_date
    :project: zebradm

.. doxygenfunction:: zdm::time::ut1_interval(std::span<double>, Time, Time)
    :project: zebradm

.. doxygenfunction:: zdm::time::ut1_interval(Time, Time, std::size_t)
    :project: zebradm

.. doxygenfunction:: zdm::time::ut1_interval(std::span<double>, std::string_view, std::string_view, std::string_view)
    :project: zebradm

.. doxygenfunction:: zdm::time::ut1_interval(std::string_view, std::string_view, std::string_view, std::size_t)
    :project: zebradm

Constants
^^^^^^^^^

.. doxygenvariable:: zdm::time::j2000_utc
    :project: zebradm

.. [1] Forecast values of DUT1 are published by IERS Bulletin A for a year at a time.
