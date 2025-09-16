import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "pyRuleAnalyzer"
copyright = "2025, GrupoCybersegurancaVirtus"
author = "GrupoCybersegurancaVirtus"
version = "1.1.0"
release = "1.1.0"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinxcontrib.bibtex",
]

napoleon_google_docstring = True
bibtex_bibfiles = ["bib.bib"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en_US"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {"titles_only": True}
