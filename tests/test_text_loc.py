from ascm import text as T, loc as L, atoms as A, tokens

def test_abbreviations_not_split():
    p = "Доза 5 мг/кг, т.е. 300 мг/сут. Далее см. рис. 3 для результата. Следующее предложение."
    s = [p[a:b] for a, b in T.split_sentences(p)]
    assert len(s) == 3 and "т.е." in s[0] and s[1].startswith("Далее см. рис.")

def test_section_number_not_split():
    p = "2.1. Жалобы и анамнез Рекомендуется сбор жалоб. Второе."
    assert len(T.split_sentences(p)) == 2

def test_loc_tolerant():
    ok = ["Уровень убедительности рекомендаций – C (уровень достоверности доказательств – 5)",
          "Уровень убедительности рекомендаций − A (уровень достоверности доказательств – 1)",
          "Уровень убедительности рекомендаций для пембролизумаба** - А (уровень достоверности доказательств - 2)",
          "Уровень убедительности рекомендаций - В (уровень достоверности доказа­тельств - 2)"]
    assert all(L.parse(x) for x in ok)
    assert L.parse("Просто текст") is None

def test_loc_fields_returned_as_written():
    assert L.parse("Уровень убедительности рекомендаций – 4 (уровень достоверности доказательств – С)") == ("4", "С")

DOC = ("Введение.\n\nРекомендуется провести МРТ. Это нужно для оценки.\n\n"
       "Уровень убедительности рекомендаций – B (уровень достоверности доказательств – 1).\n\n"
       "Следующий абзац без рекомендации.")

def test_atoms_keep_recommendation_with_loc():
    norm, S = A.build_sentences(DOC, tokens.proxy_count)
    prot, why = A.protected_boundaries(S)
    atoms = A.build_atoms(S, prot)
    loc_idx = next(i for i, s in enumerate(S) if s.loc)
    atom = next(a for a in atoms if loc_idx in a)
    assert len(atom) == 3                       # two recommendation sentences + the LoC line
    pairs = A.loc_pairs(S)
    assert len(pairs) == 1 and pairs[0]['block'] == [1, 2]
