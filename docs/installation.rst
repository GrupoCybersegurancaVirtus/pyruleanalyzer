Installation
============

From PyPI
---------

This package can be installed via pip:

.. code-block:: bash

    $ pip install pyruleanalyzer

Development Install
-------------------

To install in editable mode for development:

.. code-block:: bash

    $ git clone https://github.com/GrupoCybersegurancaVirtus/pyruleanalyzer.git
    $ cd pyruleanalyzer
    $ pip install -e .

C Extension (Optional)
----------------------

The package includes an optional C extension (``_tree_traversal.c``) for accelerated vectorized tree traversal. The ``setup.py`` build script will attempt to compile it automatically during installation. If no C compiler is available, the package will fall back to a pure NumPy implementation transparently -- no additional configuration is required.

To verify whether the C extension is active:

.. code-block:: python

    from pyruleanalyzer._accel import USE_C_EXTENSION
    print("C extension active:", USE_C_EXTENSION)

Documentation Dependencies
--------------------------

To build the documentation locally, install the optional ``docs`` extras:

.. code-block:: bash

    $ pip install -e .[docs]

This includes Sphinx, the Read the Docs theme, and sphinxcontrib-bibtex.
