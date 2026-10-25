"""Generate Sphinx Python-domain pages from ASTs without importing experiments.

Static extraction keeps documentation builds independent of OpenCV, PyTorch,
datasets, fitted models and command-line side effects. Signatures and Google
docstrings always come from the same source files as the displayed listings.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from sphinx.ext.napoleon.docstring import GoogleDocstring


def write_changed(path: Path, text: str) -> None:
    """Write generated reStructuredText only when its contents change.

    Args:
        path: Destination file in the generated reference directory.
        text: Complete UTF-8 page contents.

    Returns:
        None: Unchanged pages retain their modification times.

    Raises:
        OSError: The page cannot be read or written.
    """
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def generate(app) -> None:
    """Generate module, member and source-listing pages before Sphinx discovers files.

    Args:
        app: Sphinx application whose srcdir contains the project documentation.

    Returns:
        None: Reference pages are refreshed under doc/_generated/.

    Raises:
        OSError: Project sources or generated pages cannot be accessed.
        SyntaxError: A production Python file cannot be parsed.
    """
    root = Path(app.srcdir).parent
    generated = Path(app.srcdir) / "_generated"
    generated.mkdir(exist_ok=True)
    sources = sorted([*root.joinpath("src").glob("*.py"), *root.joinpath("tools").glob("*.py")])
    expected = {"index.rst"}
    navigation = []
    for folder, title in [("src", "Core workflow"), ("tools", "Tools and documentation build")]:
        navigation += [title, "-" * len(title), "", ".. toctree::", "   :maxdepth: 1", ""]
        for path in (p for p in sources if p.parent.name == folder):
            module = f"{folder}.{path.stem}" if path.stem != "__init__" else folder
            slug = module.replace(".", "_")
            source = path.read_text(encoding="utf-8-sig")
            tree = ast.parse(source)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            navigation.append(f"   {slug}")
            page = [module, "=" * len(module), "", f".. py:module:: {module}", "",
                    str(GoogleDocstring(ast.get_docstring(tree) or "")), "",
                    f":doc:`View source listing <source_{slug}>`", "",
                    f"Source SHA-256: ``{digest}``.", ""]
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                directive = "class" if isinstance(node, ast.ClassDef) else "function"
                signature = node.name
                if directive == "function":
                    signature += f"({ast.unparse(node.args)})"
                    if node.returns:
                        signature += " -> " + ast.unparse(node.returns)
                page += [node.name, "-" * len(node.name), "", f".. py:{directive}:: {signature}", ""]
                doc = str(GoogleDocstring(ast.get_docstring(node) or ""))
                page += ["   " + line if line else "" for line in doc.splitlines()]
                page += ["", f"   Defined in ``{path.relative_to(root).as_posix()}:{node.lineno}``.", ""]
            source_title = f"Source: {module}"
            listing = [source_title, "=" * len(source_title), "", f":doc:`Back to API <{slug}>`", "",
                       f".. literalinclude:: ../../{path.relative_to(root).as_posix()}", "   :language: python",
                       "   :linenos:", ""]
            # Source pages are in the document tree, so their links and content are checked too.
            page += [".. toctree::", "   :hidden:", "", f"   source_{slug}", ""]
            for name, lines in [(f"{slug}.rst", page), (f"source_{slug}.rst", listing)]:
                expected.add(name)
                write_changed(generated / name, "\n".join(lines))
        navigation.append("")
    title = "API reference"
    write_changed(generated / "index.rst", "\n".join([title, "=" * len(title), "",
        "These signatures, Google-style docstrings and source listings are regenerated",
        "from every production module and tool on each build. Private helpers are included",
        "for maintainers. Building the reference does not execute the matching workflow.", "", *navigation]))
    for obsolete in generated.glob("*.rst"):
        if obsolete.name not in expected:
            obsolete.unlink()


def setup(app) -> dict:
    """Register source generation for each Sphinx build.

    Args:
        app: Sphinx application under configuration.

    Returns:
        dict: Extension version and parallel-read/write safety flags.
    """
    app.connect("builder-inited", generate)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
