"""Normalisation, paragraph and sentence segmentation for Russian clinical guidelines."""
from __future__ import annotations
import re

_ABBR = (r'(?:т\.\s?е|т\.\s?д|т\.\s?п|т\.\s?к|т\.\s?н|см|рис|табл|им|др|пр|гг?|мг|мл|мкг|ст|пп?|ед|тыс|млн|'
         r'мес|нед|сут|обл|англ|лат|прим|напр|вкл|min|max|vs|Figs?|et al|approx|ECOG|N|№|в т\.\s?ч|ч|ккал|мм|кг|л|'
         r'e\.g|i\.e|Dr|Refs?|Eq|cf|ca|resp|St|Prof|Inc|Ltd|Co|No|Nos)')
_FALSE_ABBR = re.compile(r'(?:^|[\s(])' + _ABBR + r'\.$', re.I)
_FALSE_INITIAL = re.compile(r'(?:^|\s)[А-ЯЁA-Z]\.$')
_FALSE_NUM = re.compile(r'(?:^|\s)\d+(?:\.\d+)*\.$')
_SPLIT = re.compile(r'(?<=[.!?;])\s+(?=[А-ЯЁA-Z«"(\d])')
_LISTITEM = re.compile(r'^\s*(?:\d{1,2}[.)]|[-–•])\s+')
_HEADING = re.compile(r'^\s*\d+(?:\.\d+)*\.?\s+[А-ЯЁA-Z]')


def normalize(text: str) -> str:
    """Remove soft hyphens, unify minus signs and spaces. Offsets refer to the normalised text."""
    text = text.replace('­', '').replace('−', '-').replace(' ', ' ').replace('\r', '')
    return re.sub(r'[ \t]+\n', '\n', text)


def paragraphs(text: str) -> list[tuple[int, int]]:
    """Spans of paragraphs (runs without a blank line), stripped."""
    spans = []
    for m in re.finditer(r'(?:[^\n]|\n(?![ \t]*\n))+', text):
        s, e = m.start(), m.end()
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            spans.append((s, e))
    return spans


def split_sentences(par: str, balanced: bool = False) -> list[tuple[int, int]]:
    """balanced=True never splits inside unclosed (...) or [...] (needed for inline evidence markers in English)."""
    spans, start = [], 0
    for m in _SPLIT.finditer(par):
        pre = par[start:m.start()]
        if balanced and (pre.count('(') > pre.count(')') or pre.count('[') > pre.count(']')):
            continue
        if _FALSE_ABBR.search(pre) or _FALSE_INITIAL.search(pre):
            continue
        if _FALSE_NUM.search(pre) and len(pre) < 12:
            continue
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(par)))
    return [(s, e) for s, e in spans if e - s > 2]


def is_list_item(par: str) -> bool:
    return bool(_LISTITEM.match(par))


def is_heading(par: str) -> bool:
    return bool(_HEADING.match(par)) and len(par) < 160


_SEC_MULTI = re.compile(r'^\s*(\d+(?:\.\d+)+)\.?\s+[А-ЯЁA-Z]')
_SEC_TOP = re.compile(r'^\s*(\d{1,2})\.\s+(?:Краткая информация|Диагностика|Лечение|Реабилитация|Профилактика|'
                      r'Организация оказания|Дополнительная информация|Критерии оценки)', re.I)


def section_level(par: str, lang: str = 'ru') -> int:
    """Depth of a section heading at the paragraph start: '2.' -> 1, '2.1.' -> 2, '2.1.3.' -> 3; 0 if not a heading.
    Headings are often glued to the first sentence of the section, so long paragraphs are accepted for multi-level numbers."""
    m = _SEC_MULTI.match(par)
    if m:
        return m.group(1).count('.') + 1
    if lang == 'en':
        first = par.split('\n')[0].strip()
        if re.match(r'^(?:KQ|Clinical question)\s*\d+', first, re.I):
            return 2
        letters = [c for c in first if c.isalpha()]
        if len(first) < 90 and len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.8 \
                and not first.endswith('.'):
            return 1
        if len(par) < 90 and _EN_TOP.match(par):
            return 1
        return 0
    if len(par) < 120 and _SEC_TOP.match(par):
        return 1
    return 0


_EN_TOP = re.compile(r'^\s*(?:\d{1,2}\.?\s+)?(?:Introduction|Background|Methods?|Results?|Discussion|Conclusions?|'
                     r'Recommendations?|Epidemiology|Diagnosis|Treatment|Follow-up|Summary)\s*$', re.I)


def is_comment(par: str, lang: str = 'ru') -> bool:
    low = par.lstrip().lower()
    if lang == 'en':
        return low.startswith(('qualifying statement', 'rationale', 'comment', 'remarks'))
    return low.startswith('комментари')
