from setuptools import setup, find_packages

setup(
    name="deeprlearn",  
    version="0.0.1",  
    description="A reinforcement learning library for clasic and deep reinforcement learning research.",
    long_description=open("README.md").read(), 
    long_description_content_type="text/markdown", 
    author="Maximiliano Galindo",
    author_email="maximilianogalindo7@gmail.com",
    url="https://github.com/MaxGalindo150/deeprl",
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
