"""Token counters. 'proxy' is offline; 'bge-m3' uses the real tokenizer (needs transformers + network or a local copy)."""
from __future__ import annotations
import re


def proxy_count(s: str) -> int:
    return int(len(re.findall(r'\w+|[^\w\s]', s)) * 1.35)


def get_counter(name: str = "proxy"):
    if name == "proxy":
        return proxy_count
    from transformers import AutoTokenizer  # lazy import
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3" if name == "bge-m3" else name)
    return lambda s: len(tok.encode(s, add_special_tokens=False))
