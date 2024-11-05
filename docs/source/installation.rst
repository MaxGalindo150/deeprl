Installation Guide
==================

To start using DeepRL, follow these simple installation instructions.

**Basic Installation:**
-----------------------
The easiest way to install DeepRL is through `pip`. Run the following command:

.. code-block:: bash

    pip install deeprl

This will install the core dependencies required to use DeepRL.

**Optional Dependencies:**
--------------------------
For certain features or enhanced functionality, you might want to install additional packages:

- **Gymnasium**: Used for working with various reinforcement learning environments.
  
  .. code-block:: bash

      pip install gymnasium

- **PyTorch**: DeepRL relies on PyTorch for underlying computations. If you haven't installed it already, follow the `official PyTorch installation guide <https://pytorch.org/get-started/locally/>`_ for the version that suits your environment.

**Development Installation:**
-----------------------------
If you want to contribute to the development of DeepRL or modify its source code, follow these steps:

1. **Clone the repository**:

   .. code-block:: bash

       git clone https://github.com/yourusername/deeprl.git

2. **Navigate to the project directory**:

   .. code-block:: bash

       cd deeprl

3. **Create and activate a virtual environment (optional but recommended)**:

   .. code-block:: bash

       python3 -m venv venv
       source venv/bin/activate  # On Windows use `venv\Scripts\activate`

.. 4. **Install the library in editable mode with development dependencies**:

..    .. code-block:: bash

..        pip install -e .[dev]

..    This will install DeepRL in editable mode, allowing you to make changes to the codebase and immediately reflect them without reinstalling. The `[dev]` option will include additional packages for development like linters, testing frameworks, etc.

**Common Issues and Troubleshooting:**
--------------------------------------
- **Issue**: *pip install deeprl* fails with a missing dependency.
  
  **Solution**: Ensure you have Python 3.7 or higher installed. Verify your Python version with:

  .. code-block:: bash

      python --version

- **Issue**: `torch` is not installed or version conflict.

  **Solution**: Install PyTorch by following the `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_.

**Next Steps:**
---------------
Once DeepRL is installed, head over to the **Tutorials** section to start using the library and build your first RL agent!
