import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'PyRuleAnalyzer'
copyright = '2025, GrupoCybersegurancaVirtus'
author = 'GrupoCybersegurancaVirtus'

extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'pt_BR'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
