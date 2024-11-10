from setuptools import setup, find_packages

setup(
    name="deeprl",  # Nombre del paquete
    version="0.1.0",  # Versión inicial
    description="A reinforcement learning library for clasic and deep reinforcement learning research.",
    long_description=open("README.md").read(),  # LEE el README.md
    long_description_content_type="text/markdown",  # Indica que el README usa Markdown
    author="Tu Nombre",
    author_email="tuemail@ejemplo.com",
    url="https://github.com/tuusuario/deeprl",  # Repositorio del proyecto
    license="MIT",  # Licencia del proyecto
    packages=find_packages(),  # Encuentra automáticamente todos los paquetes
    include_package_data=True,  # Incluye archivos definidos en MANIFEST.in
    install_requires=[
        "gymnasium>=0.27.0",
        "torch>=1.10.0",
        "numpy>=1.21.0",
        "scikit-learn>=0.24.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",  # Versión mínima de Python
)
