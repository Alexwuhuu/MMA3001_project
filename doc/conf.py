"""Sphinx configuration for the source-derived project documentation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))

project = "MMA3001: Cross-season Apple-tree Matching"
author = "MMA3001 project contributors"
extensions = ["myst_parser", "source_api"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
html_title = project
html_static_path = ["_static"]
html_css_files = ["documentation.css"]
html_sidebars = {"**": ["about.html", "navigation.html", "relations.html", "searchbox.html"]}
html_theme_options = {"description": "Workflow, data contracts and source-derived API reference"}
html_show_sourcelink = False
html_last_updated_fmt = None
myst_heading_anchors = 3
highlight_language = "python"
