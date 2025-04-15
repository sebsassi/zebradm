# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = 'zebradm'
copyright = '2025, Sebastian Sassi'
author = 'Sebastian Sassi'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

import sys
print(sys.executable)

extensions = ['breathe']

breathe_projects = {'zebradm': '_build/xml'}
breathe_default_project = "zebradm"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_static_path = ['_static']

# -- Options for LaTeX output ------------------------------------------------

latex_engine = 'lualatex'
latex_documents = [(master_doc, 'zebradm_doc.tex', 'ZebraDM', author, 'manual', False)]
latex_toplevel_sectioning = 'section'
latex_elements = {
    'papersize': 'a4paper',
    'preamble': '',
    'extrapackages': r'\usepackage{microtype}'
}
