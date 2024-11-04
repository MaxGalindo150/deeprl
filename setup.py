# import os 
# import sys

# if sys.playform == "win32" and sys.maxsize.bit_length() == 31:
#     print(
#         "32-bit Windows Python is not supported. Please switch to 64-bit Python."
#     )
#     sys.exit(-1)

# import platform

# BUILD_LIBDEEPRL_WHL = os.getenv("BUILD_LIBDEEPRL_WHL", "0") == "1"
# BUILD_PYTHON_ONLY = os.getenv("BUILD_PYTHON_ONLY", "0") == "1"

# python_min_version = (3, 9, 0)
# python_min_version_str = ".".join(map(str, python_min_version))
# if sys.version_info < python_min_version:
#     print(
#         f"You are using Python {platform.python_version()}. Please use Python {python_min_version_str} or newer."
#     )
    
# import filecmp
# import glob
# import importlib
# import importlib.util
# import json
# import shutil
# import subprocess
# import sysconfig
# import time
# from collections import defaultdict

# import setuptools.command.build_ext
# import setuptools.command.install
# import setuptools.command.sdist
# from setuptools import Extension, find_packages, setup
# from setuptools.dist import Distribution
# # from tools.build_pytorch_libs import build_caffe2
# # from tools.generate_torch_version import get_torch_version
# # from tools.setup_helpers.cmake import CMake
# # from tools.setup_helpers.env import build_type, IS_DARWIN, IS_LINUX, IS_WINDOWS
# # from tools.setup_helpers.generate_linker_script import gen_linker_script


# setup.py
from setuptools import setup, find_packages

setup(
    name='deeprl',  # Cambia esto al nombre de tu librería
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'ipython'
        # Esto asegura que matplotlib se instale como requisito
        # Puedes agregar otros requisitos aquí
    ],
    author='Maximiliano Galindo',
    author_email='maximilianogalindo7@gmail.com',
    description='Descripción breve de tu librería',
    url='https://github.com/MaxGalindo150/DeepRL',  # URL de tu repositorio
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
