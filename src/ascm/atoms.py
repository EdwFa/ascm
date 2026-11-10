"""Build sentences, protected boundaries F and atoms for one document."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import re
from . import text as T, loc as L


@dataclass
class Sentence:
    idx: int
    para: int
    start: int
    end: int
    text: str
    loc: bool
    list_item: bool
    heading: bool
    tokens: int = 0
    sec_level: int = 0
    comment: bool = False
    loc_kind: str = ''


def build_sentences(raw: str, count_tokens, lang: str = 'ru', family: str = 'ru_uur') -> tuple[str, list[Sentence]]:
    norm = T.normalize(raw)
    sents: list[Sentence] = []
    for pi, (ps, pe) in enumerate(T.paragraphs(norm)):
        par = norm[ps:pe]
        li, hd = T.is_list_item(par), T.is_heading(par)
        lvl, com = T.section_level(par, lang), T.is_comment(par, lang)
        first = True
        for s, e in T.split_sentences(par, balanced=(lang != 'ru')):
            txt = re.sub(r'\s+', ' ', par[s:e]).strip()
            kind = L.evidence_kind(txt, family)
            sents.append(Sentence(len(sents), pi, ps + s, ps + e, txt, bool(kind), li, hd, count_tokens(txt),
                                  lvl if first else 0, com if first else False, kind))
            first = False
    return norm, sents


def protected_boundaries(S: list[Sentence]):
    """prot[j] is True if the boundary between S[j-1] and S[j] is in F. why[j] in {'loc','list','rec'}."""
    n = len(S)
    prot, why = [False] * (n + 1), [None] * (n + 1)
    for j in range(1, n):
        a, b = S[j - 1], S[j]
        if b.loc and b.loc_kind != 'inline':
            prot[j], why[j] = True, 'loc'
        elif a.list_item and b.list_item and b.para == a.para + 1 and not a.loc:
            prot[j], why[j] = True, 'list'
    for j in range(1, n):
        if S[j].loc:
            k = j if S[j].loc_kind == 'inline' else j - 1
            while k >= 1 and S[k - 1].para == S[k].para and not S[k - 1].loc:
                if not prot[k]:
                    prot[k], why[k] = True, 'rec'
                k -= 1
    return prot, why


def build_atoms(S: list[Sentence], prot: list[bool]) -> list[list[int]]:
    if not S:
        return []
    atoms, cur = [], [0]
    for j in range(1, len(S)):
        if prot[j]:
            cur.append(j)
        else:
            atoms.append(cur)
            cur = [j]
    atoms.append(cur)
    return atoms


def loc_pairs(S: list[Sentence]) -> list[dict]:
    """Ground-truth recommendation-evidence pairs.
    'sep'   : the evidence sentence and the contiguous recommendation block before it.
    'inline': the marker sentence with the earlier sentences of the same paragraph; single-sentence recommendations are
              skipped (they cannot be split)."""
    pairs = []
    for j, s in enumerate(S):
        if not s.loc:
            continue
        if s.loc_kind == 'inline':
            k = j
            while k >= 1 and S[k - 1].para == S[k].para and not S[k - 1].loc:
                k -= 1
            if k < j:
                pairs.append(dict(loc=j, adjacent=j - 1, block=list(range(k, j)), kind='inline'))
        elif j > 0 and not S[j - 1].loc:
            k = j - 1
            while k >= 1 and S[k - 1].para == S[k].para and not S[k - 1].loc:
                k -= 1
            pairs.append(dict(loc=j, adjacent=j - 1, block=list(range(k, j)), kind='sep'))
    return pairs


def to_json(norm: str, S: list[Sentence], prot, why, atoms) -> dict:
    return dict(text=norm, sentences=[asdict(s) for s in S], protected=prot, why=why, atoms=atoms)
