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

   \text{TT} = \text{UTC} + 42.184\text{s} + \text{leap seconds}.

.. [1] Forecast values of DUT1 are published by IERS Bulletin A for a year at a time.
