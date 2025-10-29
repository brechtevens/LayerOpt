Background
==========

Multilayer Dielectric Slab Structures
-------------------------------------

Assume that the ingoing medium is free space and that the characteristic impedance of the outgoing medium is given by :math:`\eta_{M+1}`
The refractive index of layer :math:`i` is

.. math::

   n(x_i) = \sqrt{\epsilon(x_i)}

and the characteristic impedance of layer :math:`i` is

.. math::

   \eta(x_i) = \eta_0 \sqrt{\frac{1}{\epsilon(x_i)}}

At each interface :math:`i`, the reflection coefficient is

.. math::

   \rho(x_{i-1}, x_i) =
   \begin{cases}
     \dfrac{\eta(x_1) - 1}{\eta(x_1)+1}, & \text{if } i=1,\\[2mm]
     \dfrac{\eta_{M+1} - \eta(x_M)}{\eta_{M+1} + \eta(x_M)}, & \text{if } i=M+1,\\[1mm]
     \dfrac{\eta(x_i) - \eta(x_{i-1})}{\eta(x_i) + \eta(x_{i-1})}, & \text{otherwise.}
   \end{cases}

Forward and backward fields propagate recursively through the structure. Let :math:`E^f_{i,+}` and :math:`E^f_{i,-}` denote the forward and backward fields at the left of interface :math:`i`. Then:

.. math::

   \begin{bmatrix} E^f_{i,+} \\ E^f_{i,-} \end{bmatrix}
   =
   \frac{1}{1+\rho(x_{i-1}, x_i)}
   \begin{bmatrix} 1 & \rho(x_{i-1}, x_i) \\ \rho(x_{i-1}, x_i) & 1 \end{bmatrix}
   \begin{bmatrix} e^{j k_0 t n(x_i)} & 0 \\ 0 & e^{-j k_0 t n(x_i)} \end{bmatrix}
   \begin{bmatrix} E^f_{i+1,+} \\ E^f_{i+1,-} \end{bmatrix}

The reflection and transmission responses are

.. math::

   \Gamma^f(x) = \frac{E^f_{1,-}}{E^f_{1,+}}, \quad
   \mathcal{T}^f(x) = \frac{E^f_{M+1,+}}{E^f_{1,+}}

The reflection, transmission, and absorption coefficients are

.. math::

   R_{\rm model}^f(x) = |\Gamma^f(x)|^2, \quad
   T_{\rm model}^f(x) = |\mathcal{T}^f(x)|^2, \quad
   A_{\rm model}^f(x) = 1 - R_{\rm model}^f(x) - T_{\rm model}^f(x)

The shielding effectiveness is

.. math::

   \text{SE}_{\rm model}^f(x) = 10 \log_{10}(\frac{1}{T_{\rm model}^f(x)})

Learning Permittivity Model
---------------------------

LayerOpt learns a mapping from material parameters to complex permittivity
and permeability using experimental or simulated data.  
The parametric model :math:`\epsilon(x_i, p)` is fit by minimizing a user-defined cost function.  
For example, one may minimize the error between predicted and measured
absorption-to-reflection ratios and shielding effectiveness. The framework allows arbitrary loss functions, so users can implement custom objectives that prioritize specific metrics, frequency bands, or robustness criteria.
See :ref:`learning_permittivity_models` for more details.

Optimizing Material Parameters
------------------------------

The goal of optimization within this python package is to design multilayer shields that maximize the worst-case ratio of absorption to reflection while satisfying a minimum shielding effectiveness requirement.  

The optimization workflow typically consists of two stages:

1. **Continuous Relaxation**  
   Solve over all valid material parameters using a global optimizer
   (e.g., basin-hopping or differential evolution). This produces high-quality
   continuous solutions efficiently.

2. **Discrete Search**  
   In case the designer is only able to manufacture a discrete set of material parameters, an additional brute-force search step is added to look for the best discrete solution near the continuous one.

This two-stage approach ensures **sample-efficient learning** while producing
high-performance multilayer sequences that meet design constraints.
See :ref:`optimizing_multilayer_structures` for more details.