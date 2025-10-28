Examples
========

This section provides example scripts demonstrating how to use the main features
of **LayerOpt**. Each example can be run directly from the ``scripts/`` folder.

.. contents::
   :local:
   :depth: 1

Permittivity Models
-------------------

**File:** ``scripts/Permittivity-example.py``

This example demonstrates how to:

1. Use built-in permittivity models such as simple linear and quadratic models.
2. Create a custom permittivity model.
3. Understand the physical sign convention for complex permittivity (ε = ε' + iε'').
4. Visualize the real (ε') and imaginary (ε'') parts of permittivity.

**Key classes used:**

- :class:`LayerOpt.Permittivity.PermittivityModel`
- :class:`LayerOpt.Permittivity.SimpleLinearPermittivityModel`
- :class:`LayerOpt.Permittivity.QuadraticPermittivityModel`

**Example code:**

.. literalinclude:: ../scripts/Permittivity-example.py
   :language: python
   :linenos:
   :caption: Example - Using and extending PermittivityModel classes

.. note::

   Numba JIT compilation of ``_compute_real_permittivity`` and ``_compute_imaginary_permittivity``` is crucial!

   If you remove ``@njit`` from ``_compute_real_permittivity`` or
   ``_compute_imaginary_permittivity``, your model will still work
   for direct evaluation, but will fail when used inside
   :class:`LayerOpt.core.MultilayerStructure`, because that class uses njit-compiled
   loops for performance.

   In short:

   - **Use ``@njit``** for permittivity models that will be simulated.
   - **Skip ``@njit``** only for pure testing or debugging.


MultilayerStructure
-------------------

**File:** ``scripts/MultilayerStructure-example.py``

This example demonstrates how to:

1. Define a multilayer dielectric structure.
2. Use a simple linear permittivity model.
3. Compute and visualize reflectance (R), transmittance (T), absorption (A),
   absorption-to-reflectance ratio (A/R), and shielding effectiveness (SET).

**Key classes used:**

- :class:`LayerOpt.core.MultilayerStructure`
- :class:`LayerOpt.Permittivity.SimpleLinearPermittivityModel`

**Example code:**

.. literalinclude:: ../scripts/MultilayerStructure-example.py
   :language: python
   :linenos:
   :caption: Example - Using the MultilayerStructure class

Note that one can use any other Permittivity model in this example script, such as

- :class:`LayerOpt.Permittivity.LinearPermittivityModel`
- :class:`LayerOpt.Permittivity.SimpleQuadraticPermittivityModel`

.. note::

   When running the code, you may see a warning such as:

   ``NumbaPerformanceWarning: np.dot() is faster on contiguous arrays, called on (Array(complex128, 2, 'A', False, aligned=True), Array(complex128, 1, 'C', False, aligned=True))``

   This is a harmless performance warning from Numba and can be safely ignored. It does not affect correctness or numerical results.

Experiments
-----------

**File:** ``scripts/Experiment-example.py``

This example demonstrates how to:

1. Load experimental multilayer measurement data.
2. Access S-parameters (S11, S21), transmittance (T), reflectance (R), and absorptance (A).
3. Plot A/R versus shielding effectiveness (SET) for different experiments.

**Key classes used:**

- :class:`LayerOpt.core.ExperimentData`

**Example code:**

.. literalinclude:: ../scripts/Experiment-example.py
   :language: python
   :linenos:
   :caption: Example - Using the ExperimentData class

Learning Permittivity Models
----------------------------

**File:** ``scripts/MultilayerLearner-example.py``

This example demonstrates how to:

1. Learn permittivity model parameters from experimental data using :class:`LayerOpt.core.MultilayerLearner` with a user-defined cost function.
2. Visualize and compare predicted versus measured transmittance (T), reflectance (R), and absorption (A).

**Key classes used:**

- :class:`LayerOpt.core.MultilayerLearner`
- :class:`LayerOpt.Permittivity.SimpleQuadraticPermittivityModel`
- :class:`LayerOpt.Experiment.ExperimentData`

**Example code:**

.. literalinclude:: ../scripts/MultilayerLearner-example.py
   :language: python
   :linenos:
   :caption: Example - Learning permittivity model parameters with MultilayerLearner

.. note::

   In this example, a custom cost function is used that only penalizes errors in R (reflectance). 
   This illustrates how the choice of cost function affects the predicted coefficients: R is accurately matched, but T and A are not. 
   To optimize across all coefficients (T, R, A) and compare, re-run the script using ``cost_fun = learner.cost_coeffs()``.

Optimizing Multilayer Structures
--------------------------------

**File:** ``scripts/MultilayerOptimizer-example.py``

This example demonstrates how to:
 
1. Optimize the material parameters of a multilayer dielectric slab structure with a given permittivity model. The goal is to minimize R/A (or equivalently to maximize A/R), while satisfying a shielding effectiveness (SE) constraint.  
2. Visualize the optimization results.

**Key classes used:**

- :class:`LayerOpt.core.MultilayerOptimizer`  
- :class:`LayerOpt.core.MultilayerStructure`  
- :class:`LayerOpt.Permittivity.SimpleLinearPermittivityModel`

**Example code:**

.. literalinclude:: ../scripts/MultilayerOptimizer-example.py
   :language: python
   :linenos:
   :caption: Example - Optimizing multilayer dielectric slab structures with MultilayerOptimizer

.. note::

   In this example, the discrete solution performs significantly worse than the continuous one, since it is restricted to material parameters listed in ``material_list``.  
   Consider adding 0.125 to the ``material_list`` and re-running the script.