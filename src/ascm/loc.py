"""Detection of the Russian evidence line: 'Уровень убедительности рекомендаций – X (уровень достоверности доказательств – N)'.
LoC = УУР (A-C), LoR = УДД (1-5)."""
from __future__ import annotations
import re

ANCHOR = re.compile(r'Уровень убедительности', re.I)
_SLOT = r'([A-ZА-Яa-z0-9])?'
PATTERN = re.compile(
    r'Уровень убедительности рекомендаци[йи](?: для [^()]{1,120}?)?\s*-?\s*' + _SLOT +
    r'\s*-?\s*\(\s*уровень достоверности д[оа]\d?казател\s?ьств\s*-?\s*' + _SLOT + r'\s*\)', re.I)


def _norm(t: str) -> str:
    t = t.replace('­', '').replace('−', '-').replace('–', '-').replace('—', '-')
    return re.sub(r'\s+', ' ', t)


def has_anchor(sentence: str) -> bool:
    return bool(ANCHOR.search(sentence))


def parse(sentence: str):
    """Return (uur, udd) or None. Fields may be swapped or corrupted in the source; they are returned as written."""
    m = PATTERN.search(_norm(sentence))
    if not m:
        return None
    return m.group(1), m.group(2)


def count_matches(text: str) -> tuple[int, int]:
    """(tolerant matches, anchors) in a whole document."""
    t = _norm(text)
    return len(PATTERN.findall(t)), len(ANCHOR.findall(t))
