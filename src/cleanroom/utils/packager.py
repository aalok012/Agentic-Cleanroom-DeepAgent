"""Assemble generated MVC code into a runnable FastAPI application package.

The Code Agent emits one file per functional requirement (a flat ``GeneratedCode`` with
``files``); for the FastAPI stack each file imports the shared handle
``from app.extensions import Base, SessionLocal, get_db`` and (for controllers/views)
DEFINES — but never registers — an ``APIRouter``. This step lays those files out as an
importable ``app`` package, adds the shared SQLAlchemy base/session plus a ``create_app()``
factory that initializes the database and registers every router it finds, and writes a
``requirements.txt``.

It is MECHANICAL only — it lays out files and writes fixed scaffolding, never program
logic — so the no-feedback isolation of the pipeline is unaffected.
"""

import ast
import re
from pathlib import Path

# Conventional layer directories (match the planner's deterministic file_path layout).
_LAYER_DIRS = ("models", "controllers", "views")
_SINGULAR_TO_DIR = {"model": "models", "view": "views", "controller": "controllers"}

_EXTENSIONS_TEMPLATE = '''"""Shared SQLAlchemy base and session factory for generated models."""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()

engine = create_engine("sqlite:///./app.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
'''

_INIT_TEMPLATE = '''"""Application factory: build the FastAPI app, init the database, register routers.

Generated mechanically by the pipeline's packager — no business logic lives here.
Run with:  uvicorn app:create_app --factory   (or: python -m app)
"""
import importlib
import pkgutil

from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.extensions import Base, engine

_LAYERS = ("models", "controllers", "views")


def create_app() -> FastAPI:
    app = FastAPI()

    # Import every generated module so models register on the metadata and routers
    # become importable, then register any APIRouter instances found. Each file is
    # generated in isolation, so router tags collide; register every distinct router
    # exactly once under a prefix made unique by its source module.
    registered_ids: set[int] = set()
    for layer in _LAYERS:
        try:
            package = importlib.import_module(f"app.{layer}")
        except ModuleNotFoundError:
            continue
        for info in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"app.{layer}.{info.name}")
            for value in vars(module).values():
                if not isinstance(value, APIRouter) or id(value) in registered_ids:
                    continue
                registered_ids.add(id(value))
                prefix = f"/{layer}/{info.name}"
                app.include_router(value, prefix=prefix, tags=[info.name])

    # Manual test console, written by the pipeline's ui_packager from the planner's
    # contracts. Optional: the app runs identically when static/ is absent.
    static = Path(__file__).parent / "static"
    if (static / "index.html").exists():
        app.mount("/static", StaticFiles(directory=str(static)), name="static")

        @app.get("/", include_in_schema=False)
        def _console() -> FileResponse:
            return FileResponse(str(static / "index.html"))
    else:
        @app.get("/", include_in_schema=False)
        def _docs() -> RedirectResponse:
            return RedirectResponse("/docs")

    Base.metadata.create_all(bind=engine)
    return app
'''

_MAIN_TEMPLATE = '''"""Run the assembled app:  python -m app"""
import uvicorn

from app import create_app

if __name__ == "__main__":
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
'''


def _layer_dir(path: str, mvc_layer: str) -> str:
    """The package subdirectory for a generated file: prefer the directory in its contract
    path (e.g. 'controllers/foo.py' -> 'controllers'), else the pluralized mvc_layer."""
    parts = Path(path).parts
    if len(parts) > 1 and parts[0] in _LAYER_DIRS:
        return parts[0]
    return _SINGULAR_TO_DIR.get((mvc_layer or "").strip().lower(), "controllers")


def _collect(generated_code: dict) -> list[tuple[str, str, str]]:
    """[(layer_dir, filename, content)] from the flat GeneratedCode ``files`` schema.

    Same-named files within a layer are disambiguated by fr_id rather than overwritten.
    Tolerates the legacy ``increments`` shape so older IRs still package.
    """
    files = generated_code.get("files") or [
        f for inc in generated_code.get("increments", []) for f in inc.get("files", [])
    ]
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for f in files:
        path = f.get("path", "")
        layer_dir = _layer_dir(path, f.get("mvc_layer", ""))
        name = Path(path).name or "module.py"
        if (layer_dir, name) in seen:
            stem, suffix = Path(name).stem, Path(name).suffix
            name = f"{stem}_{str(f.get('fr_id', 'x')).replace('.', '_')}{suffix}"
        seen.add((layer_dir, name))
        out.append((layer_dir, name, f.get("content", "")))
    return out


def _model_class_defs(content: str) -> list[ast.ClassDef]:
    """Top-level classes that subclass Base (SQLAlchemy declarative) in a generated file."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    out: list[ast.ClassDef] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and any(
            (isinstance(b, ast.Name) and b.id in ("Base", "Model"))
            or (isinstance(b, ast.Attribute) and b.attr in ("Base", "Model"))
            for b in node.bases
        ):
            out.append(node)
    return out


def _model_score(node: ast.ClassDef) -> tuple[int, int]:
    """Rank a model definition by (has_primary_key, column_count) so the most complete wins."""
    has_pk = False
    columns = 0
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and (
            (isinstance(sub.func, ast.Attribute) and sub.func.attr == "Column")
            or (isinstance(sub.func, ast.Name) and sub.func.id == "Column")
        ):
            columns += 1
            for kw in sub.keywords:
                if kw.arg == "primary_key" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    has_pk = True
    return (1 if has_pk else 0, columns)


def _dedupe_models(files: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Collapse model classes redefined across files (an isolated-generation artifact).

    Each functional requirement is generated independently, so the same SQLAlchemy model
    (e.g. ``SearchResult``) is often declared in several files — sometimes as an empty
    stub with no primary key, which makes the shared declarative base unmappable. Keep the
    single most complete definition of each model and rewrite every other declaration into
    an import of the canonical one, so the assembled app imports cleanly.
    """
    canonical: dict[str, tuple[tuple[int, int], str, str]] = {}
    for layer_dir, name, content in files:
        stem = Path(name).stem
        for node in _model_class_defs(content):
            score = _model_score(node)
            current = canonical.get(node.name)
            better = current is None or score > current[0] or (
                score == current[0] and (layer_dir == "models") and current[1] != "models"
            )
            if better:
                canonical[node.name] = (score, layer_dir, stem)

    rewritten: list[tuple[str, str, str]] = []
    for layer_dir, name, content in files:
        stem = Path(name).stem
        removals = [
            (node, canonical[node.name])
            for node in _model_class_defs(content)
            if canonical.get(node.name) and canonical[node.name][1:] != (layer_dir, stem)
        ]
        if removals:
            lines = content.splitlines()
            for node, (_, canon_layer, canon_stem) in sorted(
                removals, key=lambda r: r[0].lineno, reverse=True
            ):
                import_line = f"from app.{canon_layer}.{canon_stem} import {node.name}"
                lines[node.lineno - 1: node.end_lineno] = [import_line]
            content = "\n".join(lines) + ("\n" if content.endswith("\n") else "")
        content = _redirect_model_imports(content, stem, canonical)
        rewritten.append((layer_dir, name, content))
    return rewritten


def _redirect_model_imports(content: str, stem: str, canonical: dict) -> str:
    """Point every ``from ... import <Model>`` at the canonical module that defines it."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return content

    edits = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        model_aliases = [a for a in node.names if a.name in canonical]
        if not model_aliases:
            continue
        other = [a for a in node.names if a.name not in canonical]
        new_lines: list[str] = []
        if other and node.module:
            names = ", ".join(a.asname and f"{a.name} as {a.asname}" or a.name for a in other)
            new_lines.append(f"from {'.' * node.level}{node.module} import {names}")
        for alias in model_aliases:
            _, canon_layer, canon_stem = canonical[alias.name]
            if canon_stem == stem:
                continue
            new_lines.append(f"from app.{canon_layer}.{canon_stem} import {alias.name}")
        edits.append((node.lineno, node.end_lineno, new_lines))

    if not edits:
        return content
    lines = content.splitlines()
    for lineno, end_lineno, new_lines in sorted(edits, key=lambda e: e[0], reverse=True):
        lines[lineno - 1: end_lineno] = new_lines
    return "\n".join(lines) + ("\n" if content.endswith("\n") else "")


# The names a generated FastAPI-stack file references but, because each file is generated
# in isolation and the Code Agent trims its import list to "what it uses" (and miscounts),
# frequently forgets to import — leaving the module unimportable and taking the whole
# assembled app down at boot. Each maps to the module it is canonically imported from.
# Repairing the import is mechanical (imports only, never program logic) so isolation holds.
_KNOWN_IMPORTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("app.extensions", ("Base", "SessionLocal", "engine", "get_db")),
    ("fastapi", ("APIRouter", "Body", "HTTPException", "Depends", "status",
                 "Query", "Path", "Header", "Form")),
    ("sqlalchemy", ("Column", "Integer", "BigInteger", "SmallInteger", "String", "Text",
                    "Float", "Numeric", "Boolean", "Date", "DateTime", "Time", "ForeignKey",
                    "LargeBinary", "Enum", "JSON", "Table", "UniqueConstraint", "Index")),
    ("sqlalchemy.orm", ("Session", "relationship", "sessionmaker", "declarative_base")),
)


def _ensure_known_imports(content: str) -> str:
    """Add any well-known FastAPI/SQLAlchemy/app name a file references but never imports.

    Deterministic repair of the Code Agent's most frequent FastAPI-stack boot failures
    (``Column(Integer, ...)`` with ``Integer`` unimported, ``class X(Base)`` with ``Base``
    unimported, etc.). Merges into an existing ``from <module> import ...`` line when present,
    else inserts a new one after the last import. Touches imports only — never logic.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return content

    available: set[str] = set()           # names already importable/bound in the file
    module_nodes: dict[str, ast.ImportFrom] = {}
    last_import_line = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            last_import_line = node.end_lineno or node.lineno
            for a in node.names:
                available.add(a.asname or a.name)
            key = ("." * node.level) + (node.module or "")
            module_nodes.setdefault(key, node)
        elif isinstance(node, ast.Import):
            last_import_line = node.end_lineno or node.lineno
            for a in node.names:
                available.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            available.add(node.name)       # locally defined — don't re-import
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    available.add(t.id)

    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    lines = content.splitlines()
    new_lines: list[str] = []             # (module, names) to append at the end as fresh imports
    edits: list[tuple[int, int, str]] = []  # (start, end, replacement) for existing import lines

    for module, names in _KNOWN_IMPORTS:
        missing = [n for n in names if n in used and n not in available]
        if not missing:
            continue
        node = module_nodes.get(module)
        if node is not None:
            existing = [a.asname or a.name for a in node.names]
            edits.append((node.lineno, node.end_lineno or node.lineno,
                          f"from {module} import " + ", ".join(existing + missing)))
        else:
            new_lines.append(f"from {module} import " + ", ".join(missing))

    if not edits and not new_lines:
        return content
    for start, end, text in sorted(edits, key=lambda e: e[0], reverse=True):
        lines[start - 1: end] = [text]
    for offset, text in enumerate(new_lines):
        lines.insert(last_import_line + offset, text)
    return "\n".join(lines) + ("\n" if content.endswith("\n") else "")


def _allow_table_redefinition(content: str) -> str:
    """Let same-named tables declared across isolated files coexist on the shared metadata.

    Each FR is generated independently, so the same entity (e.g. ``orders``) is often
    declared in several files under DIFFERENT class names (``Order``, ``OrderModel``) but
    the SAME ``__tablename__``. On one declarative ``Base`` that raises
    ``InvalidRequestError: Table 'orders' is already defined``. Adding
    ``__table_args__ = {"extend_existing": True}`` to each model that declares a tablename
    (and lacks table_args) lets SQLAlchemy merge them so the app boots. Mechanical — adds a
    fixed class attribute, never business logic — so isolation is unaffected.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return content

    inserts: list[tuple[int, str]] = []  # (after_line, text)
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and any(
            (isinstance(b, ast.Name) and b.id in ("Base", "Model"))
            or (isinstance(b, ast.Attribute) and b.attr in ("Base", "Model"))
            for b in node.bases
        )):
            continue
        tablename_stmt = None
        has_table_args = False
        for stmt in node.body:
            targets = stmt.targets if isinstance(stmt, ast.Assign) else (
                [stmt.target] if isinstance(stmt, ast.AnnAssign) else [])
            for t in targets:
                if isinstance(t, ast.Name) and t.id == "__tablename__":
                    tablename_stmt = stmt
                if isinstance(t, ast.Name) and t.id == "__table_args__":
                    has_table_args = True
        if tablename_stmt is not None and not has_table_args:
            indent = " " * tablename_stmt.col_offset
            inserts.append((tablename_stmt.end_lineno or tablename_stmt.lineno,
                            f'{indent}__table_args__ = {{"extend_existing": True}}'))

    if not inserts:
        return content
    lines = content.splitlines()
    for after_line, text in sorted(inserts, key=lambda x: x[0], reverse=True):
        lines.insert(after_line, text)
    return "\n".join(lines) + ("\n" if content.endswith("\n") else "")


def _rewrite_imports(content: str, stem_to_layer: dict[str, str]) -> str:
    """Rewrite intra-project imports to absolute ``app.<layer_dir>.<module>`` paths.

    Only touches modules that correspond to a generated file stem, so the shared
    ``from app.extensions import ...`` and any third-party imports are left untouched.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return content

    lines = content.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            last = node.module.split(".")[-1]
            layer = stem_to_layer.get(last)
            if layer:
                i = node.lineno - 1
                lines[i] = lines[i].replace(f"from {node.module} import", f"from app.{layer}.{last} import", 1)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                last = alias.name.split(".")[-1]
                layer = stem_to_layer.get(last)
                if layer and alias.name != last:
                    i = node.lineno - 1
                    lines[i] = lines[i].replace(f"import {alias.name}", f"import app.{layer}.{last} as {last}", 1)

    trailing = "\n" if content.endswith("\n") else ""
    return "\n".join(lines) + trailing


def _table_name(node: ast.ClassDef) -> str | None:
    """The ``__tablename__`` string literal of a model class, if it declares one."""
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if (isinstance(t, ast.Name) and t.id == "__tablename__"
                        and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)):
                    return stmt.value.value
    return None


def _columns_of(node: ast.ClassDef, content: str) -> list[tuple[str, str, bool]]:
    """(name, ``Column(...)`` source, is_primary_key) for each column assignment in a model."""
    out: list[tuple[str, str, bool]] = []
    for stmt in node.body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Call)):
            fn = stmt.value.func
            if (isinstance(fn, ast.Name) and fn.id == "Column") or (
                    isinstance(fn, ast.Attribute) and fn.attr == "Column"):
                name = stmt.targets[0].id
                rhs = ast.get_source_segment(content, stmt.value) or "Column(String)"
                is_pk = any(
                    kw.arg == "primary_key" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                    for kw in stmt.value.keywords
                )
                out.append((name, rhs, is_pk))
    return out


def _strip_primary_key(rhs: str) -> str:
    """Remove a ``primary_key=True`` keyword from a ``Column(...)`` source, tidying commas."""
    rhs = re.sub(r"primary_key\s*=\s*True", "", rhs)
    rhs = re.sub(r"\(\s*,", "(", rhs)
    rhs = re.sub(r",\s*,", ",", rhs)
    rhs = re.sub(r",\s*\)", ")", rhs)
    return rhs


def _unify_models_by_table(files: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Collapse DIFFERENTLY-NAMED model classes that map to the SAME table into one class.

    Each FR is generated in isolation, so the same table is often modeled by classes with
    different names AND incompatible schemas — e.g. ``dishes`` declared once as
    ``DishModel(id PK, name, quantity)`` and once as ``Dish(name PK, classification)``. With
    ``extend_existing=True`` SQLAlchemy tries to merge them and dies ("Trying to redefine
    primary-key column 'name' as a non-primary-key column"), taking the whole app down at boot.
    ``_dedupe_models`` only collapses same-CLASS-NAME redefinitions, so this slips through.

    Here we pick one canonical class per table (prefer the ``models`` layer, then the richest
    definition), give it the UNION of all columns with a single primary key (``id`` if any
    definition has one, else the first declared PK; every other PK is demoted), and rewrite
    every other class for that table into an import alias of the canonical one. Mechanical —
    union of declared columns, no business logic — so isolation holds.
    """
    groups: dict[str, list[dict]] = {}
    for idx, (layer, name, content) in enumerate(files):
        stem = Path(name).stem
        for node in _model_class_defs(content):
            table = _table_name(node)
            if table:
                groups.setdefault(table, []).append(
                    {"idx": idx, "layer": layer, "stem": stem,
                     "class_name": node.name, "node": node, "content": content})

    edits_by_idx: dict[int, list[tuple[int, int, list[str]]]] = {}
    for table, group in groups.items():
        if len(group) < 2:
            continue

        union: dict[str, str] = {}            # column name -> Column(...) source (first definition wins)
        order: list[str] = []
        pk_candidates: list[str] = []
        for d in group:
            for cname, rhs, is_pk in _columns_of(d["node"], d["content"]):
                if cname not in union:
                    union[cname] = rhs
                    order.append(cname)
                if is_pk and cname not in pk_candidates:
                    pk_candidates.append(cname)
        pk_col = "id" if "id" in union else (pk_candidates[0] if pk_candidates else (order[0] if order else None))

        col_lines: list[str] = []
        for cname in order:
            rhs = union[cname]
            if cname == pk_col:
                if cname == "id":
                    rhs = "Column(Integer, primary_key=True)"
                elif "primary_key" not in rhs and rhs.endswith(")"):
                    rhs = rhs[:-1].rstrip() + ", primary_key=True)"
            else:
                rhs = _strip_primary_key(rhs)
            col_lines.append(f"    {cname} = {rhs}")

        canon = sorted(
            group,
            key=lambda d: (d["layer"] != "models", -len(_columns_of(d["node"], d["content"]))),
        )[0]
        cc, cl, cs = canon["class_name"], canon["layer"], canon["stem"]
        canon_text = [
            f"class {cc}(Base):",
            f'    __tablename__ = "{table}"',
            '    __table_args__ = {"extend_existing": True}',
        ] + (col_lines or ["    pass"])

        for d in group:
            node = d["node"]
            if d is canon:
                new = canon_text
            elif d["class_name"] == cc:
                new = [f"from app.{cl}.{cs} import {cc}"]
            else:
                new = [f"from app.{cl}.{cs} import {cc} as {d['class_name']}"]
            edits_by_idx.setdefault(d["idx"], []).append((node.lineno, node.end_lineno or node.lineno, new))

    if not edits_by_idx:
        return files
    out = list(files)
    for idx, edits in edits_by_idx.items():
        layer, name, content = files[idx]
        lines = content.splitlines()
        for start, end, new in sorted(edits, key=lambda e: e[0], reverse=True):
            lines[start - 1: end] = new
        out[idx] = (layer, name, "\n".join(lines) + ("\n" if content.endswith("\n") else ""))
    return out


def _defined_model_names(files: list[tuple[str, str, str]]) -> set[str]:
    """Every Base-subclass class name defined anywhere in the assembled package."""
    names: set[str] = set()
    for _, _, content in files:
        for node in _model_class_defs(content):
            names.add(node.name)
    return names


_STUBS_MODULE = "app.models._stubs"

# Shared handles exported by app.extensions — never mistake these for a missing model.
_EXTENSIONS_EXPORTS = frozenset({"Base", "SessionLocal", "engine", "get_db"})


def _stub_undefined_model_imports(
    files: list[tuple[str, str, str]],
) -> tuple[list[tuple[str, str, str]], str | None]:
    """Synthesize stub models for ``from app...`` imports of names defined in no file.

    Each FR is generated in isolation, so a controller frequently imports a persistence
    model (e.g. ``from app.models import Transaction``) that NO other FR was ever tasked
    with defining. That unresolved import raises ``ImportError`` at module load and takes
    the WHOLE assembled app down at boot — so a single missing model fails every endpoint,
    not just its own. For each such name we generate a minimal SQLAlchemy model (a primary
    key plus one nullable column per attribute the code reads off it) in
    ``app/models/_stubs.py`` and repoint the import there, so the app boots and every
    endpoint is judged on its own merits. Mechanical scaffolding only — fixed columns
    inferred from usage, never business logic — so isolation holds.
    """
    defined = _defined_model_names(files)
    attrs: dict[str, set[str]] = {}            # missing model -> attribute names accessed
    rewritten: list[tuple[str, str, str]] = []

    for layer_dir, name, content in files:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            rewritten.append((layer_dir, name, content))
            continue

        edits: list[tuple[int, int, list[str]]] = []
        file_missing: set[str] = set()
        for node in tree.body:
            if not (isinstance(node, ast.ImportFrom) and node.module
                    and node.module.split(".")[0] == "app" and node.level == 0
                    and node.module != "app.extensions"):
                continue
            bad = [a for a in node.names if a.name not in defined
                   and a.name not in _EXTENSIONS_EXPORTS and a.name[:1].isupper()]
            if not bad:
                continue
            file_missing.update(a.name for a in bad)
            good = [a for a in node.names if a not in bad]
            new_lines: list[str] = []
            if good:
                names_str = ", ".join(a.asname and f"{a.name} as {a.asname}" or a.name for a in good)
                new_lines.append(f"from {node.module} import {names_str}")
            stub_names = ", ".join(a.asname and f"{a.name} as {a.asname}" or a.name for a in bad)
            new_lines.append(f"from {_STUBS_MODULE} import {stub_names}")
            edits.append((node.lineno, node.end_lineno or node.lineno, new_lines))

        if file_missing:
            for sub in ast.walk(tree):
                if (isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
                        and sub.value.id in file_missing):
                    attrs.setdefault(sub.value.id, set()).add(sub.attr)
            for m in file_missing:
                attrs.setdefault(m, set())

        if edits:
            lines = content.splitlines()
            for start, end, new in sorted(edits, key=lambda e: e[0], reverse=True):
                lines[start - 1: end] = new
            content = "\n".join(lines) + ("\n" if content.endswith("\n") else "")
        rewritten.append((layer_dir, name, content))

    if not attrs:
        return rewritten, None

    blocks: list[str] = []
    for model, used in sorted(attrs.items()):
        cols = ["    id = Column(Integer, primary_key=True)"]
        for col in sorted(used):
            if col in ("id", "metadata", "query") or col.startswith("_"):
                continue
            cols.append(f"    {col} = Column(String, nullable=True)")
        blocks.append(
            f"class {model}(Base):\n"
            f'    __tablename__ = "{model.lower()}"\n'
            f'    __table_args__ = {{"extend_existing": True}}\n'
            + "\n".join(cols) + "\n"
        )
    stub_module = (
        '"""Stub models for entities imported across isolated FRs but defined in no file.\n\n'
        "Generated mechanically by the packager so the assembled app boots; columns are\n"
        'inferred from attribute usage, never from business logic."""\n'
        "from sqlalchemy import Column, Integer, String\n\n"
        "from app.extensions import Base\n\n\n"
        + "\n\n".join(blocks)
    )
    return rewritten, stub_module


def build_runnable_package(generated_code: dict, out_dir: Path, ir: dict | None = None) -> Path:
    """Assemble generated_code into out_dir/app/ as a runnable FastAPI package. Returns the app dir.

    When ``ir`` is given, also writes the static UI. With no generated frontend that is the
    manual contract console at ``app/static/index.html``; when ``ir['generated_frontend']`` is
    present the agent's per-feature pages are written alongside a deterministic index, and the
    console moves to ``console.html``. Either way it is a deliverable, never an agent input.
    """
    from src.cleanroom.utils.code_stats import code_stats

    files = _dedupe_models(_collect(generated_code))
    files = _unify_models_by_table(files)
    files, stub_module = _stub_undefined_model_imports(files)
    stem_to_layer = {Path(name).stem: layer_dir for layer_dir, name, _ in files}

    app = out_dir / "app"
    app.mkdir(parents=True, exist_ok=True)
    (app / "__init__.py").write_text(_INIT_TEMPLATE)
    (app / "extensions.py").write_text(_EXTENSIONS_TEMPLATE)
    (app / "__main__.py").write_text(_MAIN_TEMPLATE)

    layers: set[str] = set()
    for layer_dir, name, content in files:
        layers.add(layer_dir)
        dest = app / layer_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = _rewrite_imports(content, stem_to_layer)
        content = _ensure_known_imports(content)
        content = _allow_table_redefinition(content)
        dest.write_text(content)
    if stub_module is not None:
        layers.add("models")
        (app / "models").mkdir(parents=True, exist_ok=True)
        (app / "models" / "_stubs.py").write_text(stub_module)
    for layer_dir in layers:
        (app / layer_dir / "__init__.py").write_text("")

    ignore = {"app", "fastapi", "uvicorn", "sqlalchemy", "pydantic"}
    third_party = code_stats(generated_code)["third_party_libraries"]
    deps = sorted({d for d in third_party if d not in ignore} | {"fastapi", "uvicorn", "sqlalchemy"})
    (out_dir / "requirements.txt").write_text("\n".join(deps) + "\n")
    if ir is not None:
        from src.cleanroom.utils.ui_packager import build_ui

        # build_ui reads ir['generated_frontend'] itself; with no generated pages it writes
        # the deterministic console exactly as before.
        build_ui(ir, app)

    return app


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m src.cleanroom.utils.packager <full_ir.json>")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as fh:
        ir = json.load(fh)
    if "generated_code" not in ir:
        print("IR has no 'generated_code' — run the pipeline (with the LLM stages) first.")
        sys.exit(1)

    out = Path("outputs/generated") / ir.get("project_name", "project") / "runnable"
    app_dir = build_runnable_package(ir["generated_code"], out)
    print(f"Runnable FastAPI package written to: {app_dir}")
    print("Run it with:")
    print(f'  cd "{out}"')
    print("  python -m app")
