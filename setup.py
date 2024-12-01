from setuptools import setup, find_packages

setup(
    name="deeprlearn",  
    version="0.1.0",  
    description="A reinforcement learning library for clasic and deep reinforcement learning research.",
    package_data={"deeprl": ["py.typed", "version.txt"]},
    long_description=open("README.md").read(), 
    long_description_content_type="text/markdown", 
    author="Maximiliano Galindo",
    author_email="maximilianogalindo7@gmail.com",
    url="https://github.com/MaxGalindo150/deeprl",
    license="MIT",  # Licencia del proyecto
    packages=find_packages(),  # Encuentra automáticamente todos los paquetes
    include_package_data=True,  # Incluye archivos definidos en MANIFEST.in
    install_requires=[
        "gymnasium>=0.29.1,<1.1.0",
        "numpy>=1.20,<3.0",
        "torch>=2.3,<3.0",
        # For saving models
        "cloudpickle",
        # For reading logs
        "pandas",
        # Plotting learning curves
        "matplotlib",
    ],
        extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "mypy",
            # Lint code and sort imports (flake8 and isort replacement)
            "ruff>=0.3.1",
            # Reformat
            "black>=24.2.0,<25",
        ],
        "docs": [
            "sphinx>=5,<9",
            "sphinx-autobuild",
            "sphinx-rtd-theme>=1.3.0",
            # For spelling
            "sphinxcontrib.spelling",
            # Copy button for code snippets
            "sphinx_copybutton",
        ],
        "extra": [
            # For render
            "opencv-python",
            "pygame",
            # Tensorboard support
            "tensorboard>=2.9.1",
            # Checking memory taken by replay buffer
            "psutil",
            # For progress bar callback
            "tqdm",
            "rich",
            # For atari games,
            "ale-py>=0.9.0",
            "pillow",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",  # Versión mínima de Python
)
