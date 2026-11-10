from ascm import clean, loc, text as T, atoms as A, tokens

def test_cleaner_removes_watermark_and_fixes_ligatures():
    raw = ("Downloaded from ascopubs.org by 10.0.0.1 on May 1, 2025\nThe classi\ufb01cation is clear.\n\x04 First point.\n\x04 Second point.\n"
           + "Body line number one.\n" * 10 + "References\n1. Foo\n2. Bar\n")
    out = clean.clean_pdf_text(raw)
    assert "Downloaded" not in out and "10.0.0.1" not in out
    assert "classification" in out and "ﬁ" not in out
    assert "• First point." in out and "References" not in out

def test_evidence_kinds():
    assert loc.evidence_kind("Biopsy is advised [III, A].", "esmo") == "inline"
    assert loc.evidence_kind("Biopsy [I, A; ESMO-MCBS v1.1 score: 4].", "esmo") == "inline"
    assert loc.evidence_kind("(Evidence quality: Moderate; Strength of recommendation: Strong)", "asco") == "sep"
    assert loc.evidence_kind("MMC is recommended (Evidence quality: Low; Strength of recommendation: Conditional)", "asco") == "inline"
    assert loc.evidence_kind("Strength of the recommendation: Conditional for", "eras") == "sep"
    assert loc.evidence_kind("Anything [III, A]", "none") == ""

def test_balanced_split_keeps_marker_together():
    p = "MMC is recommended. (Evidence quality: Moderate; Strength of recommendation: Strong) Next sentence follows."
    s = [p[a:b] for a, b in T.split_sentences(p, balanced=True)]
    assert any("Evidence quality: Moderate; Strength of recommendation: Strong" in x for x in s)

def test_inline_marker_glues_multisentence_recommendation():
    doc = "Intro text here.\n\n• Surgery is the only cure. Its extent depends on location [II, B].\n\n• Biopsy is needed [III, A]."
    norm, S = A.build_sentences(doc, tokens.proxy_count, 'en', 'esmo')
    prot, why = A.protected_boundaries(S)
    atoms = A.build_atoms(S, prot)
    joined = [a for a in atoms if len(a) == 2]
    assert len(joined) == 1 and "extent" in S[joined[0][1]].text
    pairs = A.loc_pairs(S)
    assert len(pairs) == 1 and pairs[0]['kind'] == 'inline'    # the single-sentence recommendation is not a pair
