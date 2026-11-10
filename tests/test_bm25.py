import numpy as np
from ascm import bm25


def test_bm25_ranks_matching_text_first_and_rrf_fuses():
    texts = ["рекомендуется ПЭТ/КТ при подозрении на метастазы", "лучевая терапия на область шеи", "диета и отказ от курения"]
    s = bm25.BM25(texts).scores("метастаз ПЭТ/КТ")
    assert int(np.argmax(s)) == 0
    f = bm25.rrf([np.array([0.1, 0.9, 0.2]), np.array([1.0, 0.5, 0.0])])
    assert f.shape == (3,) and int(np.argmax(f)) in (0, 1)
