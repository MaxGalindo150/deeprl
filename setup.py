from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='deeprl',  
    version='0.1.0', 
    packages=find_packages(),  
    install_requires=[
        'numpy>=1.18.0',
        'gymnasium>=0.26.0',
        'matplotlib>=3.0.0',
        'ipython>=7.0.0',
        'pickle-mixin'
    ],
    author='Maximiliano Galindo',
    author_email='maximilianogalindo7@gmail.com',
    description='A deep reinforcement learning library based on PyTorch.',
    long_description=long_description,  
    long_description_content_type="text/markdown",  
    url='https://github.com/MaxGalindo150/DeepRL',  
    project_urls={
        "Bug Tracker": "https://github.com/MaxGalindo150/DeepRL/issues",
        "Documentation": "https://github.com/MaxGalindo150/DeepRL#readme",
    },
    classifiers=[
        'Development Status :: 3 - Alpha',  
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.9', 
    license='MIT',  
    keywords='deep reinforcement learning, AI, PyTorch, RL',
)
