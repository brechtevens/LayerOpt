.. Welcome to LayerOpt Documentation
.. =================================

.. LayerOpt is a Python package for the **data-driven design and optimization of multilayered electromagnetic interference (EMI) shields**.  
.. It enables simulation, learning, and optimization of multilayer dielectric shields.

.. Problem Setup
.. -----------------

.. Consider the design problem of a nonmagnetic multilayer dielectric structure consisting of :math:`M` layers, :math:`M+1` interfaces, and :math:`M+2` media, as visualized below.

.. .. figure:: figures/multilayer.png
..    :width: 100%
..    :align: center

.. We assume that each layer has a fixed thickness, and that the designer is able to select the material parameters :math:`x_i \in \mathbb{R}` at each layer :math:`i = 1,\dots,M`, representing for example its density or the weight fraction of a filler.

.. Then, there are two main open questions:
.. given a certain stack of materials with corresponding parameter vector :math:`x = (x_1, \dots, x_M)`, what is the expected behavior of this multilayer dielectric slab structure in terms of absorption, reflection and transmission.
.. second, how can we optimally select :math:`x = (x_1, \dots, x_M)` to achieve desired EMI shielding goals.

.. Goals
.. -----

.. LayerOpt has two main objectives:

.. 1. **Learn a permittivity model** mapping material parameters :math:`x_i` to :math:`\epsilon(x_i)` using experimental data.  
.. 2. **Optimize the material parameters** :math:`x = (x_1, \dots, x_M)` to design multilayer dielectric structures with desired EM properties.

Welcome to LayerOpt Documentation
=================================

LayerOpt is a Python package for the **data-driven design of multilayered electromagnetic interference (EMI) shields**, enabling simulation, learning, and optimization of multilayer dielectric shields.

Problem Setup
-------------

Consider the design problem of a nonmagnetic multilayer dielectric structure consisting of :math:`M` layers, :math:`M+1` interfaces, and :math:`M+2` media, as visualized below.

.. figure:: figures/multilayer.png
   :width: 100%
   :align: center

.. raw:: html

   <div style="margin-bottom: 1em;"></div>

Each layer :math:`i` has a fixed thickness, and the designer can select the material parameters :math:`x_i \in \mathbb{R}`, representing, for example, its density or the weight fraction of a filler.  
Two key questions arise:

1. Given a parameter vector :math:`x = (x_1, \dots, x_M)`, what is the expected behavior of the multilayer slab in terms of absorption, reflection, and transmission?  
2. How can we choose :math:`x = (x_1, \dots, x_M)` to meet specific EMI shielding goals?

LayerOpt addresses both of these questions by:

- **Learning a permittivity model** mapping :math:`x_i` to :math:`\epsilon(x_i)` from experimental data.  
- **Optimizing the material parameters** :math:`x` to produce multilayer structures with desired EM performance.

Quickstart
----------

LayerOpt can be installed as follows:

.. code-block:: bash

   git clone https://github.com/brechtevens/LayerOpt.git
   cd LayerOpt
   python3 -m pip install .

More details can be found on the :doc:`Installation <installation>` page.

.. toctree::
   :maxdepth: 2
   :caption: Navigation

   installation
   examples
   background
   modules

References
----------

LayerOpt builds on the optimization strategy introduced in the following work:

S. De Smedt, B. Evens, P. Ravichandran, P. Patrinos, F. Van Loock, and R. Cardinaels,  
"SMaRT Stacking: A Methodology to Produce Optimally Layered EMI Shields with Maximal Green Index Using Fused Deposition Modeling," *Adv. Funct. Mater.*, 2025, e12713.  
https://doi.org/10.1002/adfm.202512713
