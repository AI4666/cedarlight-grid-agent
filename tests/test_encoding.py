"""Every text-mode file read/write must name an encoding.

Without one, Python uses the platform locale (cp1252 on Windows), so writing the
model card's "≥" crashed run_ml.py there. CI runs on Linux (UTF-8 locale) and would
never see that failure, so this is a static check over the source instead.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ["src", "02_simulation", "03_logic", "04_uncertainty", "05_ml", "tests", "ui"]


def _mode(call: ast.Call) -> str:
    args = call.args[1:2] if isinstance(call.func, ast.Name) else call.args[0:1]
    for node in args + [k.value for k in call.keywords if k.arg == "mode"]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
    return "r"


def _missing_encoding(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name in ("read_text", "write_text"):
            text_mode = True
        elif name == "open" and (isinstance(func, ast.Name) or isinstance(func, ast.Attribute)):
            text_mode = "b" not in _mode(node)
        else:
            continue
        if text_mode and not any(k.arg == "encoding" for k in node.keywords):
            yield node.lineno, name


def test_text_io_always_names_an_encoding():
    offenders = []
    for d in SOURCE_DIRS:
        for path in sorted((ROOT / d).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            offenders += [f"{path.relative_to(ROOT)}:{line} {name}()" for line, name in _missing_encoding(tree)]
    assert not offenders, "text I/O without encoding=...:\n" + "\n".join(offenders)


def test_checker_flags_a_missing_encoding():
    bad = ast.parse('Path("x").write_text("≥")\nopen("y")\nopen("z", "rb")\nopen("w", encoding="utf-8")')
    assert [n for _, n in _missing_encoding(bad)] == ["write_text", "open"]
