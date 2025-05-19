# Using 'Sphinx' for HTML documentation on documented functions:

> pip install sphinx sphinx-autodoc-typehints  (or uv add ...)
> mkdir docs  (under my project path)
> cd docs
> sphinx-quickstart  (to initialize sphinx)

--> now answer all prompts (This creates conf.py and other files in docs)


## update conf.py
Add to 'docs/conf.py' header:
> import os
> import sys
> sys.path.insert(0, os.path.abspath(".."))


### add extensions to 'docs/conf.py':
> extensions = [
>     'sphinx.ext.autodoc',
>     'sphinx.ext.napoleon'
> ]


(Optional) Adjust Napoleon settings for your preferred docstring style:
> napoleon_google_docstring = True
> napoleon_numpy_docstring = True
> napoleon_use_param = True
> napoleon_use_rtype = True


## create a .rst file
In docs/, create a file named your_module.rst with the following content:

> Your Module
> ===========
>
> .. automodule:: your_module
>     :members:
>     :undoc-members:
>     :show-inheritance:

(under -> :members: ensures all functions/classes with docstrings are included)

### update the rst file to include my module

In docs/index.rst, add my module to the table of contents:
> .. toctree::
>    :maxdepth: 2
>    :caption: Contents:
> 
>    my_module



## Build the Documentation

From the docs/ directory, run:
> make html


## Notes:
•   For more advanced type hint support, consider adding sphinx-autodoc-typehints to your extensions.
•   You can customize the HTML theme in conf.py for a different look.
