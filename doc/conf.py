#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']

source_suffix = '.rst'

master_doc = 'index'

project = 'kmeans_smote'
copyright = '2017, Felix Last'
author = 'Felix Last'
version = '0.1.0'
release = '0.1.0'

language = 'en'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

pygments_style = 'sphinx'

todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

html_theme = 'alabaster'
html_static_path = ['_static']
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}


# -- Options for HTMLHelp output ------------------------------------------

htmlhelp_basename = 'kmeans_smotedoc'


# -- Options for LaTeX output ---------------------------------------------

latex_documents = [
    (master_doc, 'kmeans_smote.tex', 'kmeans\\_smote Documentation',
     'Felix Last', 'manual'),
]


# -- Options for manual page output ---------------------------------------

man_pages = [
    (master_doc, 'kmeans_smote', 'kmeans_smote Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

texinfo_documents = [
    (master_doc, 'kmeans_smote', 'kmeans_smote Documentation',
     author, 'kmeans_smote', 'Oversampling for imbalanced learning based on k-means and SMOTE.',
     'Miscellaneous'),
]



