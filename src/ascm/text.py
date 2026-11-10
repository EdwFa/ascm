"""Normalisation, paragraph and sentence segmentation for Russian clinical guidelines."""
from __future__ import annotations
import re

_ABBR = (r'(?:т\.\s?е|т\.\s?д|т\.\s?п|т\.\s?к|т\.\s?н|см|рис|табл|им|др|пр|гг?|мг|мл|мкг|ст|пп?|ед|тыс|млн|'
         r'мес|нед|сут|обл|англ|лат|прим|напр|вкл|min|max|vs|Fig|et al|approx|ECOG|N|№|в т\.\s?ч|ч|ккал|мм|кг|л)')
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


def split_sentences(par: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _SPLIT.finditer(par):
        pre = par[start:m.start()]
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
