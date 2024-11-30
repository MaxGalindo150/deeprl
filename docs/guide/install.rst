.. _install:

Installation
============


Prerequisites
-------------

DeepRL requires python 3.9+ and PyTorch >= 2.3

Windows
~~~~~~~

We recommend using `Anaconda <https://conda.io/docs/user-guide/install/windows.html>`_ for Windows users for easier installation of Python packages and required libraries. You need an environment with Python version 3.8 or above.

For a quick start you can move straight to installing DeepRL in the next step.

.. note::

	Trying to create Atari environments may result to vague errors related to missing DLL files and modules. This is an
	issue with atari-py package. `See this discussion for more information <https://github.com/openai/atari-py/issues/65>`_.

Stable Release
~~~~~~~~~~~~~~
To install DeepRL with pip, execute:

.. code-block:: bash

    pip install deeprl[extra]

.. note::
        Some shells such as Zsh require quotation marks around brackets, i.e. ``pip install 'deeprl[extra]'`` `More information <https://stackoverflow.com/a/30539963>`_.

This includes an optional dependencies like Tensorboard, OpenCV or ``ale-py`` to train on Atari games. If you do not need those, you can use:

.. code-block:: bash

    pip install deeprl


.. note::

  If you need to work with OpenCV on a machine without a X-server (for instance inside a docker image),
  you will need to install ``opencv-python-headless``.

Development version
-------------------

To contribute to DeepRL, with support for running tests and building the documentation.

.. code-block:: bash

    git clone https://github.com/MaxGalindo150/deeprl && cd deeprl
    pip install -e .[docs,tests,extra]
