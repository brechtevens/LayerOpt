Installation
============

Prerequisites
-------------

LayerOpt has been tested on **Python 3.10, 3.11, 3.12 and 3.13**. Make sure that one of these versions is installed on your system before proceeding.

Installing LayerOpt
-------------------

You can install LayerOpt directly from the GitHub repository:

.. code-block:: bash

   git clone https://github.com/brechtevens/LayerOpt.git
   cd LayerOpt
   python3 -m pip install .

This will install LayerOpt along with its required dependencies:

- numpy
- scipy
- pandas
- matplotlib
- numba
- pytest

Testing Your Installation
-------------------------

After installation, it is recommended to run the test suite to verify that everything works correctly. LayerOpt uses `pytest` for testing:

.. code-block:: bash

   python3 -m pip install pytest   # if pytest is not already installed
   pytest

This is the same procedure used in our continuous integration (CI) workflow, which ensures the package works on Python 3.10, 3.11, 3.12 and 3.13 and passes all tests.
More details on the CI workflow can be found here: `LayerOpt CI workflow <https://github.com/brechtevens/LayerOpt/actions/workflows/python-package.yml>`_.
