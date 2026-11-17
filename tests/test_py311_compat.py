"""CI and the Docker image run Python 3.11, so source must not use 3.12-only f-string syntax.

Python 3.12 (PEP 701) lets an f-string contain its own quote character, e.g. f'{d['k']}'.
On 3.11 that is a SyntaxError, which broke run_ml.py in CI while passing locally on 3.12.
The check needs 3.12's f-string tokens, so it runs on 3.12+; on 3.11 the import itself fails.
"""
import re
import sys
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ["src", "02_simulation", "03_logic", "04_uncertainty", "05_ml", "tests", "ui"]


def _quote(token: str) -> str:
    return re.sub(r"^[A-Za-z]*", "", token)[0]


def _reused_quotes(path: Path):
    open_quotes = []
    with open(path, encoding="utf-8") as f:
        for tok in tokenize.generate_tokens(f.readline):
            if tok.type in (tokenize.STRING, tokenize.FSTRING_START) and _quote(tok.string) in open_quotes:
                yield tok.start[0]
            if tok.type == tokenize.FSTRING_START:
                open_quotes.append(_quote(tok.string))
            elif tok.type == tokenize.FSTRING_END:
                open_quotes.pop()


@pytest.mark.skipif(sys.version_info < (3, 12), reason="needs 3.12 f-string tokenizer")
def test_no_fstring_reuses_its_own_quote():
    offenders = [f"{p.relative_to(ROOT)}:{line}"
                 for d in SOURCE_DIRS for p in sorted((ROOT / d).rglob("*.py")) for line in _reused_quotes(p)]
    assert not offenders, "3.12-only f-string syntax (breaks Python 3.11 CI/Docker):\n" + "\n".join(offenders)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="needs 3.12 f-string tokenizer")
def test_checker_flags_reused_quote(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("d = {}\nx = f'{d['k']}'\ny = f\"{d['k']}\"\n", encoding="utf-8")
    assert list(_reused_quotes(bad)) == [2]
