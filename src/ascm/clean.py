"""Cleaning of text extracted from PDF (English guidelines). Removes watermarks that contain personal data
(e.g. 'Downloaded from ... by <IP address>'), running headers and footers, fixes ligatures and bullets,
and inserts paragraph breaks before bullets, numbered recommendations and evidence lines."""
from __future__ import annotations
import re
from collections import Counter

_LIG = str.maketrans({'\ufb01': 'fi', '\ufb02': 'fl', '\ufb00': 'ff', '\ufb03': 'ffi', '\ufb04': 'ffl',
                      '\u00ad': '', '\u2212': '-', '\u00a0': ' ', '\u2002': ' ', '\u2003': ' ', '\u2009': ' '})
_DROP = re.compile(r'Downloaded from |All rights reserved|^Copyright\s*©|^©\s*20\d\d|\[CrossRef\]|\[PubMed\]|'
                   r'^https?://doi\.org/\S+$|^\s*doi:\s*\S+$', re.I)
_BULLET = re.compile(r'^[\x01-\x08\u2022\u25cf\uf0b7\u25aa]\s*')
_BREAK_BEFORE = re.compile(
    r'^(?:Recommendations?\s+\d+\.?|KQ\s*\d+\.|Clinical question\s*\d+:|Strength of the recommendation\s*:|Level of evidence\s*:|'
    r'Population \(P\)|Intervention \(I\)|Comparison \(C\)|Outcomes \(O\)|Qualifying statements?|\d+\.\d+\.\s+[A-Z])', re.I)


def _is_caps_heading(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return 4 <= len(letters) and len(line) < 90 and sum(c.isupper() for c in letters) / len(letters) > 0.8 \
        and not line.rstrip().endswith('.')


def clean_pdf_text(text: str, min_repeats: int = 6) -> str:
    text = text.replace('\x0c', '\n\n').translate(_LIG)
    lines = [l.rstrip() for l in text.split('\n')]
    cnt = Counter(l.strip() for l in lines if l.strip())
    out: list[str] = []
    for l in lines:
        s = l.strip()
        if not s:
            out.append(''); continue
        if _DROP.search(s) or (cnt[s] >= min_repeats and len(s) < 120) or re.fullmatch(r'[+\-−•\d\s]{1,3}', s):
            continue
        b = _BULLET.match(s)
        if b:
            out.append(''); out.append('• ' + s[b.end():]); continue
        if _BREAK_BEFORE.match(s) or _is_caps_heading(s):
            out.append('')
        out.append(s)
        if _is_caps_heading(s):
            out.append('')
    # drop the reference list (it is not clinical content and would be chunked as noise)
    idx = [i for i, l in enumerate(out) if re.fullmatch(r'(?:References|REFERENCES|Bibliography)', l.strip())]
    if idx and idx[-1] > 0.5 * len(out):
        out = out[:idx[-1]]
    t = '\n'.join(out)
    t = re.sub(r'(\w)-\n(?=[a-z])', r'\1', t)          # hyphenation at line end
    return re.sub(r'\n{3,}', '\n\n', t).strip() + '\n'
