# DeepRL Documentation

This directory contains the documentation for the DeepRL library. The documentation is written in Markdown format and is rendered using the Sphinx documentation generator.

## Build the Documentation

#### Install Sphinx and Theme

Execute the following commant in the project root

```bash 
pip install -e ".[docs]"
```

#### Building the Docs

In the `docs/` folder:

```bash
make html
```

If you want to build each time you make a change, you can use the following command:

```bash
sphinx-autobuild . _build/html
```
