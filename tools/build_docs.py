"""Validate source documentation and build a warning-free Sphinx HTML reference.

Usage::

    python tools/build_docs.py --check
    python tools/build_docs.py

Only Python's standard library is needed for --check. HTML builds additionally
need config/requirements-docs.txt; they never import the experiment modules.
"""
from __future__ import annotations

import argparse
import ast
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def source_files(root: Path) -> list[Path]:
    """Find the production modules and command-line tools covered by the reference.

    Args:
        root: Project directory containing src/ and tools/.

    Returns:
        Sorted Python source paths, including package initializers.
    """
    return sorted([*root.joinpath("src").glob("*.py"), *root.joinpath("tools").glob("*.py")])


def section(doc: str, title: str) -> str:
    """Read an unindented Google-style docstring section.

    Args:
        doc: Cleaned docstring from ast.get_docstring.
        title: Section name without the trailing colon.

    Returns:
        Section content, or an empty string when the section is absent.
    """
    match = re.search(rf"^{re.escape(title)}:\n(.*?)(?=^\S|\Z)", doc, re.M | re.S)
    return match.group(1) if match else ""


def check_docstrings(root: Path) -> dict[str, int]:
    """Reject missing module/API docs or parameter, return and explicit-raise drift.

    Args:
        root: Project containing the source modules and tools.

    Returns:
        Counts of documented modules, functions and classes.

    Raises:
        ValueError: A module or top-level callable fails the documentation contract.
        OSError: A source file cannot be read.
        SyntaxError: A source file is not valid Python.
    """
    errors = []
    counts = {"modules": 0, "functions": 0, "classes": 0}
    for path in source_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        label = path.relative_to(root).as_posix()
        counts["modules"] += 1
        if not ast.get_docstring(tree):
            errors.append(f"{label}: missing module docstring")
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            doc = ast.get_docstring(node) or ""
            location = f"{label}:{node.lineno} ({node.name})"
            if not doc:
                errors.append(f"{location}: missing docstring")
            if isinstance(node, ast.ClassDef):
                counts["classes"] += 1
                attributes = [n.target.id for n in node.body if isinstance(n, ast.AnnAssign)
                              and isinstance(n.target, ast.Name)]
                for name in attributes:
                    if not re.search(rf"^    {name}(?: \(.+?\))?:", section(doc, "Attributes"), re.M):
                        errors.append(f"{location}: undocumented attribute {name}")
                continue
            counts["functions"] += 1
            args = [arg.arg for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]]
            if node.args.vararg:
                args.append("*" + node.args.vararg.arg)
            if node.args.kwarg:
                args.append("**" + node.args.kwarg.arg)
            documented = re.findall(r"^    (\*{0,2}\w+)(?: \(.+?\))?:", section(doc, "Args"), re.M)
            if documented != args:
                errors.append(f"{location}: Args {documented} differ from signature {args}")
            if not section(doc, "Returns").strip():
                errors.append(f"{location}: missing Returns section (use None for side effects)")
            # Include worker validation errors that propagate through the enclosing API.
            body = ast.Module(body=node.body, type_ignores=[])
            for raised in ast.walk(body):
                if isinstance(raised, ast.Raise) and isinstance(raised.exc, ast.Call):
                    exception = ast.unparse(raised.exc.func)
                    if exception not in section(doc, "Raises"):
                        errors.append(f"{location}: undocumented explicit exception {exception}")
    if errors:
        raise ValueError("Documentation checks failed:\n" + "\n".join(errors))
    return counts


class LocalLinks(HTMLParser):
    """Collect HTML anchors and internal href/src references for offline validation.

    Attributes:
        anchors (set[str]): IDs and legacy named anchors found in the document.
        targets (list[str]): Referenced href/src values, including external URLs.
    """

    def __init__(self):
        """Initialize empty link and anchor collections.

        Returns:
            None: The parser is ready to consume generated HTML.
        """
        super().__init__(convert_charrefs=True)
        self.anchors = set()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        """Collect references and anchors from one HTML start tag.

        Args:
            tag: Lowercase HTML element name supplied by HTMLParser.
            attrs: Parsed name/value attribute pairs for the element.

        Returns:
            None: Reference and anchor collections are updated in place.
        """
        values = dict(attrs)
        if "id" in values:
            self.anchors.add(values["id"])
        if tag == "a" and "name" in values:
            self.anchors.add(values["name"])
        self.targets.extend(value for key, value in attrs if key in {"href", "src"} and value)


def check_html(folder: Path) -> int:
    """Check all generated relative links, local assets and HTML fragments.

    Args:
        folder: Sphinx HTML output directory.

    Returns:
        Number of HTML documents checked, including search/index pages.

    Raises:
        ValueError: A local target, fragment or the documentation home page is missing.
        OSError: A generated page cannot be read.
    """
    pages = {}
    for path in folder.rglob("*.html"):
        parser = LocalLinks()
        parser.feed(path.read_text(encoding="utf-8"))
        pages[path.resolve()] = parser
    if (folder / "index.html").resolve() not in pages:
        raise ValueError("HTML build did not produce index.html")
    errors = []
    for path, parser in pages.items():
        for target in parser.targets:
            parts = urlsplit(target)
            if parts.scheme or parts.netloc:
                continue
            resolved = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            if resolved.is_dir():
                resolved /= "index.html"
            if not resolved.is_file():
                errors.append(f"{path.name}: missing target {target}")
            elif parts.fragment and resolved in pages and unquote(parts.fragment) not in pages[resolved].anchors:
                errors.append(f"{path.name}: missing fragment {target}")
    if errors:
        raise ValueError("HTML link checks failed:\n" + "\n".join(errors))
    return len(pages)


def main() -> None:
    """Check source docs, then optionally build and validate the HTML site.

    Returns:
        None: Counts and the generated home-page path are printed.

    Raises:
        SystemExit: CLI arguments are invalid, Sphinx is unavailable or its build fails.
        ValueError: Source documentation or generated local links fail validation.
        OSError: Source files or generated documentation cannot be accessed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check source documentation without Sphinx")
    args = parser.parse_args()
    print("Source documentation:", check_docstrings(ROOT))
    if args.check:
        return
    try:
        from sphinx.cmd.build import build_main
    except ImportError as exc:
        raise SystemExit("Install config/requirements-docs.txt before building HTML") from exc
    output = ROOT / "doc/_build/html"
    # A full warning-as-error build catches outdated references even after source edits.
    status = build_main(["-q", "-b", "html", "-E", "-a", "-W", "--keep-going", str(ROOT / "doc"), str(output)])
    if status:
        raise SystemExit(status)
    print(f"Validated {check_html(output)} HTML pages: {output / 'index.html'}")


if __name__ == "__main__":
    main()
