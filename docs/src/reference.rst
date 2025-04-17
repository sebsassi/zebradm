Library reference
=================

Zernike-based Radon transforms
------------------------------

Types
^^^^^

.. doxygenclass:: zdm::zebra::IsotropicAngleIntegrator
    :project: zebradm

.. doxygenclass:: zdm::zebra::AnisotropicAngleIntegrator
    :project: zebradm

.. doxygenclass:: zdm::zebra::IsotropicTransverseAngleIntegrator
    :project: zebradm

.. doxygenclass:: zdm::zebra::AnisotropicTransverseAngleIntegrator
    :project: zebradm

Functions
^^^^^^^^^

.. doxygenfunction:: zdm::zebra::radon_transfomr
    :project: zebradm

.. doxygenfunction:: zdm::zebra::radon_transform_inplace
    :project: zebradm

Containers and views
--------------------

Types
^^^^^

.. doxygenclass:: zdm::SHExpansionVector
    :project: zebradm

.. doxygenclass:: zdm::SuperSpan
    :project: zebradm

.. doxygenclass:: zdm::MultiSuperSpan
    :project: zebradm

Type aliases
^^^^^^^^^^^^

.. doxygentypedef:: zdm::SHExpansion
    :project: zebradm

.. doxygentypedef:: zdm::ZernikeExpansion
    :project: zebradm

.. doxygentypedef:: zdm::SHExpansionSpan
    :project: zebradm

.. doxygentypedef:: zdm::SHExpansionVectorSpan
    :project: zebradm

.. doxygentypedef:: zdm::ZernikeExpansionSpan
    :project: zebradm

Coordinate transforms
---------------------

Functions
^^^^^^^^^

.. doxygenfunction:: zdm::coordinates::cartesian_to_spherical_geo
    :project: zebradm

.. doxygenfunction:: zdm::coordinates::spherical_to_cartesian_geo
    :project: zebradm

.. doxygenfunction:: zdm::coordinates::cartesian_to_spherical_phys
    :project: zebradm

.. doxygenfunction:: zdm::coordinates::spherical_to_cartesian_phys
    :project: zebradm
