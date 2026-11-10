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


# ---- evidence families -------------------------------------------------------------------------------------------
# kind 'sep'   : the evidence statement starts its own sentence (Russian УУР/УДД line, ASCO '(Evidence quality: ...)', ERAS lines)
# kind 'inline': the marker sits inside the recommendation sentence (ESMO '[III, A]')
_ESMO = re.compile(r'\[\s*[IV]{1,3}\s*,\s*[A-E]\s*(?:;[^\]]{0,80})?\]')
_ASCO = re.compile(r'\(\s*(?:Type\s*:[^;)]{0,80};\s*)?Evidence\s+quality\s*:\s*[A-Za-z ,/-]+;\s*Strength\s+of\s+recommendation\s*:\s*[A-Za-z ,/-]+\)', re.I)
# GRADE-style (ATA 2025, EAES/SAGES): '(Strong recommendation, Moderate certainty evidence)', '(Conditional recommendation based on low certainty of evidence)'
_GRADE = re.compile(r'\(\s*(?:Strong|Conditional|Weak)\s+recommendation\s*(?:,|based\s+on)', re.I)  # opening only: two-column PDFs often cut the closing part
_ERAS = re.compile(r'(?:Strength of the recommendation|Level of evidence)\s*:', re.I)
FAMILIES = {'ru_uur': None, 'esmo': _ESMO, 'asco': _ASCO, 'eras': _ERAS, 'grade': _GRADE, 'none': None}


def evidence_kind(sentence: str, family: str) -> str:
    """'' if the sentence carries no evidence statement, otherwise 'sep' or 'inline'."""
    if family == 'none':
        return ''
    if family == 'ru_uur':
        return 'sep' if ANCHOR.search(sentence) else ''
    m = FAMILIES[family].search(sentence)
    if not m:
        return ''
    return 'sep' if m.start() <= 2 else 'inline'
