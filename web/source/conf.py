# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# To build the documentation to html:
# > sphinx-build -b html web\source web\build\html

import os
import sys
from datetime import datetime, timezone
_now = datetime.now().strftime("%B %d, %Y, %H:%M:%S")
local_timezone = datetime.now(timezone.utc).astimezone().tzinfo

sys.path.insert(0, os.path.abspath('./'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PHYEM'
copyright = '2023, Yi Zhang, Ramy Rashad, Andrea Brugnoli, Stefano Stramigioli'
author = 'RaM, EEMCS, University of Twente'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",  # use Jupyter notebooks in Sphinx.
    "sphinx.ext.todo",
]

todo_include_todos = True

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_title = "PHYEM"
html_help_basename = 'PHYEM'
html_logo = '_static/logo-light.png'
html_favicon = '_static/favicon.png'

html_theme_options = {
    "announcement": """
        <p style='color:white;'> &#127867 PHYEM is coming</p>
    """,
    "logo": {
        # "alt_text": "foo",
        # "text": "My awesome documentation",
        "image_light": "_static/logo-light.png",
        "image_dark": "_static/logo-dark.png",
    },
    "icon_links": [
        {
            # Label for this link
            "name": "mathischeap",
            # URL where the link will redirect
            "url": "https://www.mathischeap.com/",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "_static/favicon-mic.png",
            # The type of image to be used (see below for details)
            "type": "local",
        },
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/mathischeap/phyem",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        },
        {
            # Label for this link
            "name": "Netlify Status",
            # URL where the link will redirect
            "url": "https://app.netlify.com/sites/phyem/deploys",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "https://api.netlify.com/api/v1/badges/6a559326-c54e-4a8f-a79a-a715648c73c2/deploy-status",
            # The type of image to be used (see below for details)
            "type": "url",
        }
    ],
    "icon_links_label": "Quick Links",
    "repository_url": "https://github.com/mathischeap/phyem",
    "path_to_docs": "web/source",
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "home_page_in_toc": True,
    "show_navbar_depth": 1,
    "show_toc_level": 1,
    # "navbar_end": ["mybutton.html"],
    "extra_footer": f"<div>Last updated on {_now}, {local_timezone}</div>",
    "toc_title": "On this page",
}
