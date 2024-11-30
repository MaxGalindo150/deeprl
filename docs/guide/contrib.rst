.. _contrib::

Contributing to deeprl
======================

We are excited that you are interested in contributing to deeprl! Contributions from the community help improve and expand the project. Whether you're reporting a bug, suggesting a new feature, or contributing code, your input is highly valued.

**How to Contribute:**
----------------------
- **Report Bugs**: If you find a bug, please report it by creating an issue on our `GitHub issues page <https://github.com/MaxGalindo150/deeprl/issues>`_. Include as much detail as possible, such as steps to reproduce the issue, your system configuration, and screenshots if applicable.
- **Suggest New Features**: Have an idea for a new feature? Open an issue on GitHub to discuss it. We encourage detailed proposals that include potential use cases and implementation details.

**Setting Up Your Development Environment:**
--------------------------------------------
1. **Fork the repository**:
   Visit the `deeprl GitHub repository <https://github.com/MaxGalindo150/deeprl>`_ and click on "Fork" to create your own copy.

2. **Clone the repository**:
   Clone your forked repository to your local machine:

   .. code-block:: bash

       git clone https://github.com/MaxGalindo150/deeprl.git
       cd deeprl

3. **Create a virtual environment**:
   We recommend using a virtual environment to manage dependencies:

   .. code-block:: bash

       python3 -m venv venv
       source venv/bin/activate  # On Windows use `venv\Scripts\activate`

4. **Install the dependencies**:
   Install the required dependencies and development tools:

   .. code-block:: bash

       pip install -e .[docs,tests,extra]

**Coding Standards:**
---------------------
To maintain code quality, please follow these guidelines:

- **PEP 8**: Ensure that your code follows the Python PEP 8 style guide.

- **Type hints**: Use type annotations to improve code readability and maintenance.

- **Docstrings**: Document all public classes and methods using Google or NumPy style docstrings.

**Running Tests:**
------------------
Before submitting your changes, make sure that all tests pass:

.. code-block:: bash

    pytest tests/

If you are adding a new feature, include corresponding tests in the `tests/` directory.

**Submitting a Pull Request:**
------------------------------
1. **Create a new branch**:
   Always create a new branch for your work:

   .. code-block:: bash

       git checkout -b feature/my-new-feature

2. **Make your changes**:
   Ensure your changes follow the coding standards mentioned above.

3. **Commit your changes**:
   Write clear and descriptive commit messages:

   .. code-block:: bash

       git add .
       git commit -m "Add feature: description of the feature"

4. **Push your changes**:
   Push your branch to your forked repository:

   .. code-block:: bash

       git push origin feature/my-new-feature

5. **Open a Pull Request**:
   Go to the original repository and click on "New Pull Request." Follow the prompts to submit your PR for review.

**Review Process:**
-------------------
Once you submit a pull request, one of the maintainers will review your code. Be prepared to make changes based on feedback. The review process aims to ensure code quality and maintainability.

**Thank You!**
--------------
Thank you for considering contributing to deeprl! Your contributions make the project better for everyone.
